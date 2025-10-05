### Backpropagation Feature Contribution

: Validating input contribution with Backpropagation in ANNs 

**📌 Project Overview**

In this project, we implement and validate a backward approach to compute the influence of each input feature on the final output. 

This method stands in contrast to the forward method implemented in the previous research.

**📚 Previous Research**

- **Paper:** [Explaining Neural Networks using Input Feature Contributions](https://peerj.com/articles/cs-2802/)  
- **GitHub Repository:** [NNexplainer](https://github.com/dkumango/NNexplainer.git)

**🎯 Objectives**
1. Implement a backward approach to estimate input feature contributions.
2. Indicate how much each input feature contributes to the final output.
3. Validate that the results are consistent with the contribution estimates obtained from the forward method in previous research.

---
## 📦 다운로드 및 설치

**1. 레포지토리 클론**
```bash
git clone https://github.com/yrc00/SPADE.git
cd BFC
```

**2. 가상환경 설정 (권장)**
```python
conda create -n BFC python=3.10
conda activate BFC
pip install -r requirements.txt
```

**3. 코드 실행**
```python
python BFC_execution.py
```
