#!/bin/bash --login
#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃                 GPT MODEL SETTINGS                    ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model / Architecture settings                        ┃
# ┃ ---------------------------------------------------- ┃
# ┃ GPT-3 models use 2K sequence length/context window   ┃
# ┃ The "GPT-3 XXX" below are configs from GPT-3 paper   ┃
# ┃ https://arxiv.org/abs/2005.14165, choose based on    ┃
# ┃ your desired model size or build your own configs    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

declare -A A_NLAYERS
declare -A A_HIDDEN
declare -A A_ATEN_HEADS

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3 Small:  125M ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
MODEL_125M_KEY="GPT125M"
A_NLAYERS[$MODEL_125M_KEY]=12
A_HIDDEN[$MODEL_125M_KEY]=768
A_ATEN_HEADS[$MODEL_125M_KEY]=16

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ BERT: 1.2B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="BERT1.2B"
# NLAYERS=24
# HIDDEN=2048
# ATEN_HEADS=128

BERT_1_2B_KEY="BERT1.2B"
A_NLAYERS[$BERT_1_2B_KEY]=24
A_HIDDEN[$BERT_1_2B_KEY]=2048
A_ATEN_HEADS[$BERT_1_2B_KEY]=128

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1536
# ATEN_HEADS=24

MODEL_1_5B_KEY="GPT1_5B"
A_NLAYERS[$MODEL_1_5B_KEY]=48
A_HIDDEN[$MODEL_1_5B_KEY]=1536
A_ATEN_HEADS[$MODEL_1_5B_KEY]=24

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1600
# ATEN_HEADS=25

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 2.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="2.7B"
# NLAYERS=32
# HIDDEN=2560
# ATEN_HEADS=32

MODEL_2_7B_KEY="GPT2_7B"
A_NLAYERS[$MODEL_2_7B_KEY]=32
A_HIDDEN[$MODEL_2_7B_KEY]=2560
A_ATEN_HEADS[$MODEL_2_7B_KEY]=32

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 6.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="6.7B"
# NLAYERS=32
# HIDDEN=4096
# ATEN_HEADS=32

MODEL_6_7B_KEY="GPT6_7B"
A_NLAYERS[$MODEL_6_7B_KEY]=32
A_HIDDEN[$MODEL_6_7B_KEY]=4096
A_ATEN_HEADS[$MODEL_6_7B_KEY]=32

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 13B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="13B"
# NLAYERS=40
# HIDDEN=5120
# ATEN_HEADS=40

MODEL_13B_KEY="GPT13B"
A_NLAYERS[$MODEL_13B_KEY]=40
A_HIDDEN[$MODEL_13B_KEY]=5120
A_ATEN_HEADS[$MODEL_13B_KEY]=64

# ┏━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 18.4B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="18.4B"
# NLAYERS=40
# HIDDEN=6144
# ATEN_HEADS=48

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 20B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="20B"
# NLAYERS=44
# HIDDEN=6144
# ATEN_HEADS=64

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 25B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="25B"
# NLAYERS=64
# ------------
# HIDDEN=5760  # DEFAULT (no flash attn)
# ATEN_HEADS=64
# ------------
# HIDDEN=5888    # headdim = 5888 / 46 = 128
# ATEN_HEADS=46
# -----------------
# -- FLASH ATTN --
# headdim = 5760 / 80 = 72
# HIDDEN=5760
# ATEN_HEADS=80
# ------------

MODEL_25B_KEY="GPT25B"
A_NLAYERS[$MODEL_25B_KEY]=64
A_HIDDEN[$MODEL_25B_KEY]=6144
A_ATEN_HEADS[$MODEL_25B_KEY]=64

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 30B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="30B"
# NLAYERS=64
# HIDDEN=6144
# ATEN_HEADS=64

# head size must be divisible by 8 (requirements of flash attention)
# head num must be divisible by sequence/tensor parallel size
MODEL_30B_KEY="GPT30B"
A_NLAYERS[$MODEL_30B_KEY]=64
A_HIDDEN[$MODEL_30B_KEY]=6144
A_ATEN_HEADS[$MODEL_30B_KEY]=64

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 33B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="33B"
# NLAYERS=80                                                                                                                                                                                                  
# HIDDEN=5760                                                                                                                                                                                                      
# ATEN_HEADS=80

# MODEL_33B_KEY="GPT33B"
# A_NLAYERS[$MODEL_33B_KEY]=80
# A_HIDDEN[$MODEL_33B_KEY]=5760
# A_ATEN_HEADS[$MODEL_33B_KEY]=80

MODEL_33B_KEY="GPT33B"
A_NLAYERS[$MODEL_33B_KEY]=80
A_HIDDEN[$MODEL_33B_KEY]=6144
A_ATEN_HEADS[$MODEL_33B_KEY]=64

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 145B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="145B"
# NLAYERS=80
# HIDDEN=12288
# ATEN_HEADS=96
#
GPT145B_HIDDEN=12288
GPT145B_ATEN_HEADS=96

MODEL_145B_2L_KEY="GPT145B_2L"
A_NLAYERS[$MODEL_145B_2L_KEY]=2
A_HIDDEN[$MODEL_145B_2L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_2L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_4L_KEY="GPT145B_4L"
A_NLAYERS[$MODEL_145B_4L_KEY]=4
A_HIDDEN[$MODEL_145B_4L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_4L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_5L_KEY="GPT145B_5L"
A_NLAYERS[$MODEL_145B_5L_KEY]=5
A_HIDDEN[$MODEL_145B_5L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_5L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_6L_KEY="GPT145B_6L"
A_NLAYERS[$MODEL_145B_6L_KEY]=6
A_HIDDEN[$MODEL_145B_6L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_6L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_8L_KEY="GPT145B_8L"
A_NLAYERS[$MODEL_145B_8L_KEY]=8
A_HIDDEN[$MODEL_145B_8L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_8L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_10L_KEY="GPT145B_10L"
A_NLAYERS[$MODEL_145B_10L_KEY]=10
A_HIDDEN[$MODEL_145B_10L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_10L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_12L_KEY="GPT145B_12L"
A_NLAYERS[$MODEL_145B_12L_KEY]=12
A_HIDDEN[$MODEL_145B_12L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_12L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_16L_KEY="GPT145B_16L"
A_NLAYERS[$MODEL_145B_16L_KEY]=16
A_HIDDEN[$MODEL_145B_16L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_16L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_24L_KEY="GPT145B_24L"
A_NLAYERS[$MODEL_145B_24L_KEY]=24
A_HIDDEN[$MODEL_145B_24L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_24L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_32L_KEY="GPT145B_32L"
A_NLAYERS[$MODEL_145B_32L_KEY]=32
A_HIDDEN[$MODEL_145B_32L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_32L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_48L_KEY="GPT145B_48L"
A_NLAYERS[$MODEL_145B_48L_KEY]=48
A_HIDDEN[$MODEL_145B_48L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_48L_KEY]="${GPT145B_ATEN_HEADS}"

MODEL_145B_64L_KEY="GPT145B_64L"
A_NLAYERS[$MODEL_145B_64L_KEY]=64
A_HIDDEN[$MODEL_145B_64L_KEY]="${GPT145B_HIDDEN}"
A_ATEN_HEADS[$MODEL_145B_64L_KEY]="${GPT145B_ATEN_HEADS}"

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 175B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="175B"
# NLAYERS=96
# HIDDEN=12288
# ATEN_HEADS=96
# if [ -z "$NLAYERS" ]; then
#   A_NLAYERS[$MODEL_145B_KEY]="${NLAYERS}"
#   echo "Caught NLAYERS=${NLAYERS} from env, using this value!"
# else
#   A_NLAYERS[$MODEL_145B_KEY]=80
#   echo "Using default NLAYERS=80"
# fi
MODEL_145B_KEY="GPT145B"
A_NLAYERS[$MODEL_145B_KEY]=80
A_HIDDEN[$MODEL_145B_KEY]=12288
A_ATEN_HEADS[$MODEL_145B_KEY]=96

MODEL_1T_1L_KEY="GPT1T_1L"
A_NLAYERS[$MODEL_1T_1L_KEY]=1
A_HIDDEN[$MODEL_1T_1L_KEY]=25600
A_ATEN_HEADS[$MODEL_1T_1L_KEY]=160

MODEL_1T_2L_KEY="GPT1T_2L"
A_NLAYERS[$MODEL_1T_2L_KEY]=2
A_HIDDEN[$MODEL_1T_2L_KEY]=25600
A_ATEN_HEADS[$MODEL_1T_2L_KEY]=160

MODEL_1T_4L_KEY="GPT1T_4L"
A_NLAYERS[$MODEL_1T_4L_KEY]=4
A_HIDDEN[$MODEL_1T_4L_KEY]=25600
A_ATEN_HEADS[$MODEL_1T_4L_KEY]=160


MODEL_1T_8L_KEY="GPT1T_8L"
A_NLAYERS[$MODEL_1T_8L_KEY]=8
A_HIDDEN[$MODEL_1T_8L_KEY]=25600
A_ATEN_HEADS[$MODEL_1T_8L_KEY]=160


export MODEL_SIZE="${MODEL_SIZE_KEY}"
export NLAYERS="${A_NLAYERS[$MODEL_SIZE_KEY]}"
export HIDDEN="${A_HIDDEN[$MODEL_SIZE_KEY]}"
export ATEN_HEADS="${A_ATEN_HEADS[$MODEL_SIZE_KEY]}"
