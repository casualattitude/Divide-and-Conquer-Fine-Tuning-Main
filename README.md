# Divide-and-Conquer-Fine-Tuning-Main

## Abstract

Medical image analysis tasks often suffer
from insufficient training data. Pre-trained models contain
abundant prior information about feature representation
or pattern discrimination. Fine-tuning a pre-trained model
on medical tasks using task-specific data improves the
performance of task models. However, current fine-tuning
methods focus more on reducing computational cost than
promoting the model performance. In addition, there is still
no conclusion about vital problems such as ”how many
parameters should be fine-tuned” and ”which group of
parameters should be fine-tuned”, which is important to
achieving optimal performance. In this paper, we define
the parameters whose removal incurs large loss increase
of downstream task model as fundamental parameters. We
conduct extensive investigations and observe that roughly
10% parameters of a pre-trained model are fundamental
to various downstream tasks, and these fundamental pa-
rameters dominate the optimal convergence of pre-trained
model on downstream tasks. According to these observa-
tions, we assume that the fundamental parameters repre-
sent task-agnostic but important information that should
not be overly changed while the remaining parameters
should be updated sufficiently during fine-tuning. Based on
the assumption, we propose a divide-and-conquer method
for optimal adaption of the pre-trained parameters to down-
stream tasks. We learn and apply a penalty factor to the
gradients of the fundamental parameters to control the
updates of these parameters, and fine-tune the remaining
parameters normally. Our method leads to obvious perfor-
mance improvement over full-parameter and other state-of-
the-art fine-tuning methods on four medical image classifi-
cation tasks with various amount of training data, providing
new perspectives to pre-trained model adaptation.
## Usage

### Prune

1. **Data Preparation**: 
   - Make sure your dataset is properly organized into folders. The dataset should be divided into `train` and `test` directories.
   - The images should be labeled and organized by classes (e.g., `negative`, `positive`).

2. **Running the prune Step**:
- Execute the script to prepare your dataset by running:
  ```bash
  bash prune/cap/research/run_gradual_pruning.sh 
  ```
- After the pruning process is complete, the pruned model will be saved in the `models/` directory.

### Finetuning

1. **Model Setup**:
   - You can load a pre-trained model (e.g., MAE)  on your dataset.
   - Ensure you have set the following arguments:
     - `data_path1` and `data_path2` to specify your training and validation data paths.
     - `spweight` to specify the sparse weight file.
     - `weights` to point to the pre-trained model's path.

2. **Fine-Tuning**:
   - Once the model and data are prepared, the fine-tuning can be performed with the following command:
     ```bash
     python train2.py --data_path1 <path_to_train_data> --data_path2 <path_to_val_data> --weights <path_to_pretrained_model> --batch_size <batch_size> --epochs <epochs>
     ```

## Requirements

- Python 3.9
- PyTorch
- torchvision
- TensorBoard

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

