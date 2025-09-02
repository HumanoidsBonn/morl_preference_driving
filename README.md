# MORL Preference Driving
Code repository for **"Multi-Objective Reinforcement Learning for Adaptable Personalized Autonomous Driving"** – Surmann et al., ECMR 2025

**Multi-Objective Reinforcement Learning for Adaptable Personalized Autonomous Driving**  
by **Hendrik Surmann**  
Supervised by **Jorge de Heuvel** and **Prof. Dr. Maren Bennewitz**  
University of Bonn, 2025

## 📌 Overview

This project extends the **Stable Baselines3 TD3 implementation** to develop a **multi-objective reinforcement learning (MORL) framework** that incorporates **dynamic user preferences** in autonomous driving scenarios. The implementation is based on the **PD-MORL algorithm** ([GitHub](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm)) and **Stable-Baselines3** ([GitHub](https://github.com/DLR-RM/stable-baselines3/tree/v2.0.0/stable_baselines3/td3)) and is evaluated using the **CARLA simulator (v0.9.15)**.

The goal is to train **a single policy network** capable of dynamically balancing multiple driving objectives, such as:
- **Efficiency**
- **Comfort**
- **Aggressiveness**
- **Speed**

Our focus is on training a single policy network capable of realizing multiple driving styles from vision-based input.
The model is evaluated in diverse urban driving scenarios to assess its ability to align driving behavior with user preferences.
Advanced traffic rules are only partially considered, as the primary objective is preference-adaptive driving behavior rather than raw autonomous driving performance.
While the implementation is based on the SB3 TD3 framework, we extended it to the PD-MORL algorithm (an integration of preferences into TD3 with multiple Q-values).
We further adapted PD-MORL to include a non-preference dimension. See the training function for details.

## 🚀 Installation

The installation details can be found in `install/install.txt`.  
The core dependencies include:
- **Stable-Baselines3 v2.0.0**
- **Python 3.8.10**
- **PyTorch**
- **CUDA 12.2**
- **CARLA 0.9.15**
- **Conda (for environment management)**

### 🔧 Basic Setup (short)

Ensure you have the required dependencies installed:

```bash
conda env create -f environment.yml
conda activate my_env
```
WandB might need and api-key for experiment tracking:
and wandb login <your-api-key>

## 📂 Project Structure

- **`install/`** → Contains installation instructions.
- **`sb3/`** → The main implementation of the extended TD3 MORL algorithm.
- **`run/`** → Stores trained networks/agents.
- **`scenarios/`** → Images/Videos of the implemented driving scenarios.
- **`sb3/logs/`** → Contains logging files and images and plots of the experiment results.

The training progress and results are saved in: `run/<experiment_name>_bestPref.zip/`

## 🎮 Running Experiments

### 1️⃣ Start the CARLA Server
```bash
./CarlaUE4.sh -RenderOffScreen -world-port="$ARG1" &
```

### 2️⃣ Run the MORL Agent
```bash
python td3_main.py --run="$ARG0" --client_port="$ARG1" --tm_port="$ARG2"
```
Where:
- **`ARG0`** → Name of the agent.
- **`ARG1`** → CARLA client port.
- **`ARG2`** → Traffic manager port.

Example:
```bash
python td3_main.py --run=Agent --client_port=2000 --tm_port=8000
```

### 🔧 Configurations
Modify the **config file** to adjust key settings like:

- **Enable visualization**: `SPECATE=False`
- **Train/Evaluate model**: `evaluate=False`
- **Show policy**: `showPolicy=False`
- **Traing Phase**:`key_steps= int(1e6) # or 0`

## 📝 Citation

```bibtex
@INPROCEEDINGS{Surmann2025MultiObjectiveRL,
  author={H. Surmann and J. de Heuvel and M. Bennewitz},
  title={Multi-Objective Reinforcement Learning for Adaptable Personalized Autonomous Driving},
  booktitle={Proc. European Conference on Mobile Robots (ECMR)},
  year={2025},
  address={Padua, Italy}
}
```

## 🔗 References

- **Stable Baselines3**: [stable-baselines3.readthedocs.io](https://stable-baselines3.readthedocs.io/)
- **PD-MORL Algorithm**: [GitHub Repository](https://github.com/tbasaklar/PDMORL-Preference-Driven-Multi-Objective-Reinforcement-Learning-Algorithm)
- [Supplemental Paper Video on YouTube](https://www.youtube.com/watch?v=2brpyC_edHw&ab_channel=HumanoidsBonn)


## 🔧 Support

For questions or issues, please contact:
📧 **Hendrik Surmann** - [hendrik.surmann@uni-bonn.de]
