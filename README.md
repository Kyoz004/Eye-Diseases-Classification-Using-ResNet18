<h1>Eye Diseases Classification Using ResNet18 👁️ </h1>

<p>This project focuses on classifying eye disease images using the ResNet18 deep learning architecture. The dataset contains various types of eye disease images, and the goal is to predict the disease category for each image.</p>

<h2>Dataset</h2>

<p>Download the dataset from Kaggle:</p>

<pre><code>kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification</code></pre>

<p><a href="https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/data">Kaggle Dataset: Eye Diseases Classification</a></p>

<h2>Environment Setup</h2>

<p>To set up the environment and install necessary dependencies, follow these steps:</p>

<ol>
<li>Install Jupyter and ipywidgets for interactive notebook functionality:</li>
<pre><code>pip install ipywidgets
pip install --upgrade jupyter ipywidgets</code></pre>

<li>Install the required PyTorch libraries and additional utilities:</li>
<pre><code>pip install torchmetrics
pip install torch-summary</code></pre>
</ol>

<h2>Cleaning GPU Memory</h2>

<p>To ensure optimal GPU performance, you may need to clear the cache periodically. Use the following commands:</p>

<ul>
<li>For CPU cache clearing:</li>
<pre><code>sudo sync; echo 3 &gt; /proc/sys/vm/drop_caches</code></pre>

<li>For PyTorch GPU cache clearing:</li>
<pre><code>torch.cuda.empty_cache()</code></pre>
</ul>

<h2>Notebooks</h2>

<p>You can refer to the following Kaggle notebooks for detailed implementation of eye disease classification:</p>

<ul>
<li><a href="https://www.kaggle.com/code/gpiosenka/2-models-f1-scores-73-and-94">Notebook 1: 2 Models F1 Scores 73 and 94</a></li>
<li><a href="https://www.kaggle.com/code/faizalkarim/pytorch-eye-disease-classification-93-7/notebook">Notebook 2 (Main): PyTorch Eye Disease Classification 93.7%</a></li>
</ul>

<h2>Model Architecture</h2>

<p>We use the ResNet18 model architecture, which is a deep convolutional neural network designed for image classification tasks. It consists of:</p>

<ul>
<li>Residual learning for deeper networks</li>
<li>18 layers with identity mappings</li>
<li>Convolutional layers for feature extraction</li>
</ul>

<h2>Results</h2>

<p>The trained model achieves an accuracy of up to 90.7%, after using Resnet18 with CNN.</p>

<h2>License</h2>

<p>Please refer to the dataset's licensing information on Kaggle for any restrictions or usage policies.</p>
