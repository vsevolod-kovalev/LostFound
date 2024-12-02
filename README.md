# resnet18
<img width="1149" alt="Pasted Graphic 33" src="https://github.com/user-attachments/assets/be169b3e-c2a2-4d4a-8a12-ec016d91ae3a">
<p>ResNet50 began to memorize the dataset leading to overfitting. Validation loss is not decreasing.</p>
<img width="931" alt="Pasted Graphic 31" src="https://github.com/user-attachments/assets/e06c306c-c5cd-4597-88ef-fad808af5622">
<p>A larger batch size (64 vs. 32) results in slower training per epoch but faster convergence.</p>
<img width="896" alt="Pasted Graphic 34" src="https://github.com/user-attachments/assets/8815ca83-1d4f-4388-bd01-ede09c64c5d8">
<p>Validation and training losses are at a plateau. Writing a scheduler to gradually decrease the learning rate.</p>


