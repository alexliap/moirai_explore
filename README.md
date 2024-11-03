# Thesis Stuff/Notes/TODOs

## ✅ TODO

- [x] Find univariate dataset for first tests

- [x] Finetune MOIRAI on this dataset in order to beat on train and validation

- [x] Find a univariate dataset of a subdomain of the first one and test thesis hypothesis
  - [x] Make iterative train loop
  - [x] Test 3 different configurations for finetuning:
    - Use model of training iteration N-1 for training iteration N. 1 backward pass.
    - At each training iteration use the Stage 1 model for finetuning. 1 backward pass.
    - At each training iteration use the Stage 1 model for finetuning. Multiple backward passes.
      - Without dropout
      - With 10% dropout
  - [ ] For Stage 2, also perform iterative training of the pretrained model in order to test the hypothesis
  - [x] Experiment with different ways of evaluation/visualization
    - [x] Get forecasts and targets, in order to calculate whichever metric

