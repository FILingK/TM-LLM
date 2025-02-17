import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from layers.Embed import DataEmbedding, DataEmbedding_wo_time
from peft import LoraConfig, get_peft_model


class FlattenHead(nn.Module):
    def __init__(self, d_model, Tstep):
        super().__init__()
        self.Tstep = Tstep
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(384, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, self.Tstep),
        )

    def forward(self, x):
        B, N, d_model = x.shape
        x = x.reshape(B * N, d_model)

        x = self.mlp(x)
        x = x.reshape(B, N, self.Tstep)
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff

        self.configs = configs

        self.flattenhead = FlattenHead(configs.d_model, self.configs.enc_in)

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed,
                                           configs.freq,
                                           configs.dropout)  # modify***
        if configs.llm_model == 'gpt2':
            self.llm_model = GPT2Model.from_pretrained('../GPT2'
                                                       , output_attentions=True, output_hidden_states=True)
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
            self.tokenizer = AutoTokenizer.from_pretrained(
                '../GPT2',
                trust_remote_code=True,
                local_files_only=True
            )
        elif configs.llm_model == 'deepseek_R1':
            # Load model directly
            self.llm_config = AutoConfig.from_pretrained('../deepseek_R1_1.5b')
            self.llm_config.num_hidden_layers = configs.gpt_layers
            self.llm_config.output_attentions = True
            self.llm_config.output_hidden_states = True
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                '../deepseek_R1_1.5b',
                trust_remote_code=True,
                local_files_only=True,  # If the model file exists locally, load it from the local storage.
                config=self.llm_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                '../deepseek_R1_1.5b',
                trust_remote_code=True,
                local_files_only=True
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for i, (name, param) in enumerate(self.llm_model.named_parameters()):
            if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.llm_model.to(device=device)

        if self.task_name == 'imputation':
            self.ln_proj = nn.LayerNorm(configs.d_model)
            self.out_layer = nn.Linear(
                configs.d_model,
                configs.c_out,
                bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]

        return None

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        B, L, M = x_enc.shape
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,d_model]
        # geant
        prompt = (
            f"<|start_prompt|>"
            f'###discription:###'
            f"The dataset consists of OD flow pairs between 23 routers (v1 to v23), "
            f"with a total of 529 OD flows (x1 to x529), each row of the data represents an OD pair. Each OD pair has a time step length of {self.seq_len}."
            f"Each flow xij represents the volume of traffic between two routers in the network, where xij is the flow from router vi to router vj, with i,j∈{1, 2, ..., 23}"
            f"The flow x1 is the flow from v1 to v1, x2 is the flow from v1 to v2, x23 is the flow from v1 to v23 ,x24 is the flow from v2 to v1 "
            f"and so on, up to x529, representing the flow from v23 to v23. "
            f"The flow index k for any given xij is given by the formula：k = (i - 1)*23 + j where i,j∈{1, 2, ..., 23}"
            f"Some values in the dataset are missing, and the missing values are marked as 0. "
            f'###task:###'
            f"The task is to learn the temporal relationships within the complete data of each row, "
            f"as well as the traffic relationships between different rows, to infer and fill in the missing data"
            f"Finally, only return the completed dataset after filling in the missing values."
            f"<|<end_prompt>|>"
        )
        # abilene
        # prompt = (
        #     f"<|start_prompt|>"
        #     f'###discription:###'
        #     f"The dataset consists of OD flow pairs between 12 routers (v1 to v12), "
        #     f"with a total of 144 OD flows (x1 to x144), each row of the data represents an OD pair. Each OD pair has a time step length of {self.seq_len}."
        #     f"Each flow xij represents the volume of traffic between two routers in the network, where xij is the flow from router vi to router vj, with i,j∈{1, 2, ..., 12}"
        #     f"The flow x1 is the flow from v1 to v1, x2 is the flow from v1 to v2, x12 is the flow from v1 to v12 ,x13 is the flow from v2 to v1 "
        #     f"and so on, up to x144, representing the flow from v12 to v12. "
        #     f"The flow index k for any given xij is given by the formula：k = (i - 1)*12 + j where i,j∈{1, 2, ..., 12}"
        #     f"Some values in the dataset are missing, and the missing values are marked as 0. "
        #     f'###task:###'
        #     f"The task is to learn the temporal relationships within the complete data of each row, "
        #     f"as well as the traffic relationships between different rows, to infer and fill in the missing data"
        #     f"Finally, only return the completed dataset after filling in the missing values."
        #     f"<|<end_prompt>|>"
        # )
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))
        prompt_embeddings = prompt_embeddings.expand(x_enc.size(0), -1, -1)
        combined_input = torch.cat((prompt_embeddings, enc_out), dim=1)  # Concatenate the prompt and the data
        if self.configs.llm_model == "gpt2":
            outputs = self.llm_model(inputs_embeds=combined_input).last_hidden_state

        else:
            outputs = self.llm_model(inputs_embeds=combined_input).hidden_states[-1]



        outputs = self.ln_proj(outputs[:, -self.configs.c_out:, :])
        dec_out = self.flattenhead(outputs).permute(0, 2, 1)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.seq_len, 1))
        return dec_out
