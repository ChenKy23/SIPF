# SIPF
The respository for **Self-Iterative Process Feedback for Small Language Models' Reasoning**

# Environment Setup

This outlines the steps required to set up the SIPF environment with specific CUDA and PyTorch versions, along with additional installation requirements.

## Environment Requirements

- **CUDA Version:** 12.1
- **PyTorch Version:** 2.3.0

## Setup Instructions

Follow these steps to configure the environment:

### 1. Create the SIPF Environment

Create the SIPF environment with the required versions of PyTorch, CUDA, and additional dependencies using the following command:

```bash
conda create --name sipf pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 xformers -c pytorch -c nvidia -c xformers
```

### 2. Activate the Environment

Activate the newly created environment with:

```bash
conda activate sipf
```

### 3. Install a Specific Version of Unsloth (Optional)

Due to compatibility issues with the latest version of Unsloth, it is recommended to install a specific version using:

```bash
pip install unsloth@git+https://github.com/unslothai/unsloth.git@933d9fe2cb2459f949ee2250e90a5b610d277ea
```

### 4. Install Required Libraries

Install the required libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 5. Install Pass@k Evaluation Library

Navigate to the `human_eval_master` folder and install the pass@k evaluation library:

```bash
cd human_eval_master
pip install -e .
```

## Additional Notes

- Ensure your system meets the necessary hardware and software requirements for CUDA 12.1 and PyTorch 2.3.0.
- The specific version of Unsloth is optional but recommended.
- For further customization or troubleshooting, refer to the official documentation of the respective packages.

---