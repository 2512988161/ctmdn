# **CT-MDN and Baseline Models**

This repository provides implementations and experiments for **CT-MDN** and several continuous-time and discrete-time sequence models, built upon and extended from the **Liquid Time-Constant Networks (LTC)** framework.

## **1. Preparation**

Please follow the official preparation instructions from the original LTC repository to set up the environment and dependencies:

👉 https://github.com/raminmh/liquid_time_constant_networks#preparation

This includes installing the required Python packages and preparing the datasets used in the experiments.

## **2. Training Models**

The training script supports multiple model architectures, including both baseline methods and our proposed continuous-time memory models.

### **Supported Models**

You can specify the model type using the --model argument:

```
--model {lstm, ltc, node, ctgru, ctrnn, ctmdn, ctmdnx, ctmdnxadapt, selfattention, ctmdnselfattention, nmodectmdnxadapt, cfcctmdn, nmodectmdn, nmodeltcctmdn, nmodeltcmixctmdn, cfcgatectmdn}
```

- **lstm**: Standard LSTM
- **ltc**: Liquid Time-Constant Network
- **node**: Neural ODE-based model
- **ctgru**: Continuous-Time GRU
- **ctrnn**: Continuous-Time RNN
- ctmdn / ctmdnx / ctmdnxadapt / nmodectmdnxadapt / cfcctmdn / nmodectmdn / nmodeltcctmdn / nmodeltcmixctmdn / cfcgatectmdn: Proposed Continuous-Time Memory Dynamic Network variants
- **selfattention**: Discrete-time self-attention baseline
- **ctmdnselfattention**: Continuous-time self-attention with memory dynamics

### **Example Training Command**

```
python exps/train_all.py \
  --model ctmdnx \
  --task person \
  --seq_len 32 \
  --size 32 \
  --epochs 200 \
  --unfolds 6 
```

## **3. Experimental Protocol**

All experiments follow the **same experimental design, data preprocessing, and evaluation protocols** as introduced in the original **LTC paper**, unless otherwise specified.

This ensures a fair and consistent comparison across all baseline models and the proposed CT-MDN variants.
