# SIPF
The respository for **Self-Iterative Process Feedback for Small Language Models' Reasoning**

# SIPF Environment Setup

This guide provides the steps required to set up the SIPF environment with specific versions of CUDA and PyTorch.

## Environment Requirements

- **CUDA Version:** 12.1
- **PyTorch Version:** 2.3.0

## Setup Instructions

Follow these steps to configure the environment:

### 1. Create the SIPF Environment

To create the SIPF environment with the required versions of PyTorch and CUDA, use the following command:

```bash
conda create --name sipf pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 xformers -c pytorch -c nvidia -c xformers
```

### 2. Activate the Environment

Activate the newly created environment with:

```bash
conda activate sipf
```

### 3. Install Specific Version of Unsloth (Optional)

Due to compatibility issues with the latest version of Unsloth, it is recommended to install a specific version using:

```bash
pip install unsloth@git+https://github.com/unslothai/unsloth.git@933d9fe2cb2459f949ee2250e90a5b610d277ea
```

### 4. Install Required Libraries

Install the required libraries listed in the `requirements.txt` file by running:

```bash
pip install -r requirements.txt
```

## Additional Notes

- Ensure that your system meets the necessary hardware and software requirements for CUDA 12.1 and PyTorch 2.3.0.
- The specific version of Unsloth is optional but recommended if you encounter compatibility issues.
- For further customization or troubleshooting, refer to the official documentation of the respective packages.

---

This setup guide helps ensure that the SIPF environment is configured correctly for optimal performance with the specified dependencies.
```

Feel free to adjust or expand on any sections to better suit your project needs!