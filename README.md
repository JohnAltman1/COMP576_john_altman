This model is a modification of the CGCNN from:\
\
Xie, T., & Grossman, J. C. (2018). Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Physical Review Letters, 120(14), 145301. https://doi.org/10.1103/PhysRevLett.120.145301
 \
 \
It is designed to determine if two elements in a structure are forming an alloy with each other.
To run the pretrained model on the prediction data, navigate to the COMP576_final_proj folder and use this command:
'''bash
python predict.py model_best.pth.tar COMP576_final_proj/data/predictdata  
'''bash
To train the model yourself, use this command:
'''
python main.py data/mydata --task classification --lr 0.01 --optim Adam --n-conv 4 --n-h 3
'''
