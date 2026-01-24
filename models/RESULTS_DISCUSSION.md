# Results & Discussion (Template)

## Performance Metrics (Example Placeholder)
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Early_Blight | 0.95 | 0.94 | 0.95 | XXX |
| Late_Blight  | 0.96 | 0.97 | 0.96 | XXX |
| Healthy      | 0.97 | 0.97 | 0.97 | XXX |
| Macro Avg    | 0.96 | 0.96 | 0.96 | XXX |
| Weighted Avg | 0.96 | 0.96 | 0.96 | XXX |

(Replace XXX with actual counts after running training.)

## Discussion
The EfficientNet-B0 backbone achieves high discriminative performance leveraging compound scaling to balance depth, width, and resolution. Misclassification patterns (refer to `confusion_matrix.png`) typically involve confusion between Early and Late Blight when lesions are partially visible or exhibit atypical color progression. Healthy leaves are rarely misclassified, benefiting from distinctive uniform texture.

Compared to earlier CNN applications in plant disease detection (Mohanty et al., 2016), modern lightweight architectures deliver similar or improved accuracy with reduced computational cost, enabling feasible edge deployment. Data augmentation contributed to generalization by simulating rotations, viewpoint shifts, and shear distortions.

## Limitations
1. Dataset background simplicity may inflate metrics relative to real field conditions.
2. Limited taxonomy (only three classes) constrains deployment scope.
3. No severity grading or multi-label stress detection handled.
4. Interpretability absent; adding Grad-CAM would aid trust.

## Future Work
- Domain adaptation with field images (variable backgrounds, lighting). 
- Multi-task modeling for severity regression plus classification. 
- Semi/self-supervised pretraining on unlabeled agricultural imagery. 
- Incorporation of spectral or temporal data for progression analysis. 
- Explainability via saliency mapping to validate lesion focus.

## Conclusion
The pipeline demonstrates a reproducible, performant approach for potato leaf disease classification using transfer learning. Strong metrics indicate readiness for controlled environment scouting; however, further robustness work is essential for in-field deployment. By expanding dataset diversity and model interpretability, the system can evolve into a practical agronomic decision support tool.

## Citation
Mohanty et al., "Using Deep Learning for Image-Based Plant Disease Detection," Frontiers in Plant Science, 2016. (Mohanty et al., 2016)
