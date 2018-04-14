# textgenrnn

Generate text using a pretrained neural network with a few lines of code, or easily train your own text-generating neural network of any size and complexity.

textgenrnn is a Python 3 module on top of [Keras](https://github.com/fchollet/keras)/[TensorFlow](https://www.tensorflow.org) for generating text using recurrent neural networks, with many cool features:

* A modern neural network architecture which utilizes new techniques as attention-weighting and skip-embedding to accelerate training and improve model quality.
* Able to train the model using contextual labels, allowing it to learn much faster and produce better results.
* Able to configure RNN size, the number of RNN layers, and whether to use bidirectional RNNs.
* Able to train models on a GPU and use them with a CPU.
* Able to utilize a powerful CuDNN implementation of RNNs when trained on the GPU, which massively speeds up training time as opposed to normal LSTM implementations.

## Examples

```python
from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.generate()
```
```
The confirmed the #Cubs to the Seattle Community and the most of the world of the support and the star show and a show in the series of the president and the control and have a stranger of a star to
```

The model can easily be trained on new texts, and can generate appropriate text *even after a single pass of the input data*.

```python
textgen.train_from_file('hacker-news-2000.txt', num_epochs=1)
textgen.generate(5)
```

The model weights are very small (845 KB on disk), and they can easily be saved and loaded into a new textgenrnn instance. As a result, you can play with models which have been trained on hundreds of passes through the data.

```python
textgen_2 = textgenrnn('hacker_news_500_epochs.hdf5')
textgen_2.generate(5)
```
```
Show HN: My super projects at my startup

How to start a linux project

Show HN: My mode to care

Ask HN: What is it?

A Starts Science [pdf]
```

## Usage

textgenrnn can be installed [from pypi](https://pypi.python.org/pypi/textgenrnn) via `pip`:

```
pip3 install textgenrnn
```

You can view a demo of common features in [this Jupyter Notebook](/docs/textgenrnn-demo.ipynb). (full documentation coming soon)

`/datasets` contains example datasets using Hacker News/Reddit data for training textgenrnn.

`/weights` contains further-pretrained models on the aforementioned datasets which can be loaded into textgenrnn.

`/outputs` contains examples of text generated from the above pretrained models.

## Neural Network Architecture and Implementation

![](/docs/model_shapes.png)

textgenrnn is based off of the [char-rnn](https://github.com/karpathy/char-rnn) project by [Andrej Karpathy](https://twitter.com/karpathy) with a few optimizations, such as the ability to work with very small text sequences.

The included pretrained-model follows a [neural network architecture](https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/model_def.py) inspired by [DeepMoji](https://github.com/bfelbo/DeepMoji). For the default model, textgenrnn takes in an input of up to 40 characters, converts each character to a 100-D character embedding vector, and feeds those into a 128-cell long-short-term-memory (LSTM) recurrent layer. Those outputs are then fed into *another* 128-cell LSTM. All three layers are then fed into an Attention layer to weight the most important temporal features and average them together (and since the embeddings + 1st LSTM are skip-connected into the attention layer, the model updates can backpropagate to them more easily and prevent vanishing gradients). That output is mapped to probabilities for up to [394 different characters](/textgenrnn/textgenrnn_vocab.json) that they are the next character in the sequence, including uppercase characters, lowercase, punctuation, and emoji. (if training a new model on a new dataset, all of the numeric parameters above can be configured)

Alternatively, if context labels are provided with each text document, the model can be trained in a contextual mode, where the model learns the text *given the context* so the recurrent layers learn the *decontextualized* language. The text-only path can piggy-back off the decontextualized layers; in all, this results in much faster training and better quantitative and qualitative model performance than just training the model gien the text alone.

The model weights included with the package are trained on hundreds of thousands of text documents from Reddit submissions ([via BigQuery](http://minimaxir.com/2015/10/reddit-bigquery/)), from a very *diverse* variety of subreddits. The network was also trained using the decontextual approach noted above in order to both improve training performance and mitigate authorial bias.

When fine-tuning the model on a new dataset of texts using textgenrnn, all layers are retrained. However, since the original pretrained network has a much more robust "knowledge" initially, the new textgenrnn trains faster and more accurately in the end, and can potentially learn new relationships not present in the original dataset (e.g. the [pretrained character embeddings](http://minimaxir.com/2017/04/char-embeddings/) include the context for the character for all possible types of modern internet grammar).

Additionally, the retraining is done with a momentum-based optimizer and a linearly decaying learning rate, both of which prevent exploding gradients and makes it much less likely that the model diverges after training for a long time.

## Notes

* **RESULTS WILL VARY GREATLY BETWEEN DATASETS**. Because the pretrained neural network is relatively small (128 cells), it cannot store as much data as RNNs typically flaunted in blog posts. For best results, use a dataset with atleast 2,000-5,000 documents. If a dataset is smaller, you'll need to train it for longer by setting `num_epochs` higher when calling a training method. Even then, there is currently no good heuristic for determining a "good" model.

* textgenrnn is less effective when training on/predicting longer sequences (> 200 characters). Likewise textgenrnn is less effective when training on/predicting texts with very disparate grammatical styles. If a source dataset has *both*, it may lead to unexpected results.

* A GPU is not required to retrain textgenrnn, but it will take much longer to train on a CPU. If you do use a GPU, I recommend increasing the `batch_size` parameter for better hardware utilization.

## Future Plans for textgenrnn

* More formal documentation

* A web-based implementation using tensorflow.js (works especially well due to the network's small size)

* A way to visualize the attention-layer outputs to see how the network "learns."

* A mode to allow the model architecture to be used for chatbot conversations (may be released as a separate project)

* A larger pretrained network which can accommodate longer character sequences and a more indepth understanding of language, creating better generated sentences. This may be released as a commercial product instead, if any venture capitalists are interested.

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## Credits

Andrej Karpathy for the original proposal of the char-rnn via the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## License

MIT

Attention-layer code used from [DeepMoji](https://github.com/bfelbo/DeepMoji) (MIT Licensd)