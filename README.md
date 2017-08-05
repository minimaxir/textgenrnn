# textgenrnn

textgenrnn is a Python module on top of Keras/TensorFlow which can easily generate text using a pretrained recurrent neural network on :

```python
import textgenrnn

textgen = textgenrnn()
textgen.generate()
```
```
The confirmed the #Cubs to the Seattle Community and the most of the world of the support and the star show and a show in the series of the president and the control and have a stranger of a star to
```

And can easily be trained on new texts:

```python
textgen.train_from_file('hacker-news-top-2000.txt', num_epochs=10)
textgen.generate(5)
```
```
The Google Constitutional Secret Congrats Apple Source Server's HN: What is the supering source interview

The Hacker Control Control Computer Computer Interview

Show HN: A care of a second the startup for a biggest programming to the programming to and I want to be a startup for a blown and started for the comments

The Hacker Computer Constitutional Secret to Control Is Auto of The World Programming Apple to Startup

A developers and company for a state of the support
```

The model weights are extremely small (845 KB on disk), and they can easily be saved and loaded into a new textgenrnn instance.

```python
textgen.save('hn_weights.hdf5')

textgen_2 = textgenrnn('hn_weights.hdf5')
textgen_2.generate()
```
```
Show HN: How to Control Control Boston – A Google Card
```

## Usage

You can view a demo of common features in this Jupyter Notebook. (full documentation coming soon)

`/datasets` contains example datasets for training textgenrnn.

`/weights` contains further-pretrained models on Hacker News/Reddits which can be loaded into textgenrnn.

`/output` contancs examples of text generated from the above pretrained models.

## Neural Network Architecture

![](/docs/model_shapes.png)

textgenrnn is based off of the [char-rnn](https://github.com/karpathy/char-rnn) project by [Andrej Karpathy](https://twitter.com/karpathy) with a few optimizations, such as the ability to work with smaller text sequences, and the ability for the network to learn both the start and the end of a given text sequence.

textgenrnn takes in an input of up to 40 characters, converts each character to a 100D character embedding vector, and feeds those into a 128-cell long-short-term-memory layer. That output is mapped to probabilities for up to 346 different characters that they are the next character in the sequence, including uppercase, lowercase, punctuation, and emoji.

The model weights included with the package are trained on hundreds of thousands of text documents, from Reddit submissions ([via BigQuery](http://minimaxir.com/2015/10/reddit-bigquery/)) and Facebook Pages ([via my Facebook Page Post Scraper](https://github.com/minimaxir/facebook-page-post-scraper), from a very *diverse* variety of subreddits/Pages. The network was also trained in such a way that the `rnn` is decontextualized in order to both improve training performance and mitigate authorial bias.

When fine-tuning the model on a new dataset of texts, the Embeddings layer is frozen while the `rnn` and `output` layers are retrained. However, since the original pretrained network has a much wider breath of knowledge initially, the new textgenrnn trains faster and more accurately in the end, and can learn a few new relationships potentially not present in the original dataset (e.g. the [pretrained character embeddings](http://minimaxir.com/2017/04/char-embeddings/) include the context for the character for all possible types of modern internet grammar).

Additionally, the retraining is done with a momentum-based optimizer and a linearly decaying learning rate, both of which together prevent exploding gradients and makes it much less likely that the model diverges after training for a long time.

## Notes

* **RESULTS WILL VARY GREATLY BETWEEN DATASETS**. Because the RNN is relatively small (128 cells), it cannot store as much data as other publicized RNNs. For best results, use a dataset with 2,000-5,000 documents. If a dataset is smaller, you'll need to train it for longer by setting `num_epochs` higher when calling a training method. Even then, there is currently no good heuristic for determining a "good" model, and I hope to update this module with a more consistent training regimen.

* textgenrnn is less effective when training on/predicting longer sequences (> 200 characters). Likewise textgenrnn is less effective when training on/predicting texts with very disparate grammatical styles. If a source dataset has *both*, it may lead to unexpected results.

* A GPU is not required to retrain textgenrnn, but it may take awhile to train on a CPU. If you do use a GPU, I recommend increasing the `batch_size` for better hardware utilization. Additionally, I recommend using the CNTK backend for Keras, as it [trains recurrent neural networks much faster](http://minimaxir.com/2017/06/keras-cntk/) than TensorFlow.

## Future Plans for textgenrnn

* A web-based implementation using Keras.js (works especially well due to the network's small size)

* A way to visualize the outputs of the RNN's neurons to see how the network "learns."

* A larger pretrained network which can accommodate longer character sequences and a more indepth understanding of language, creating better generated sentences. This may be released as a commercial product instead, if any venture capitalists are interested.

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## Credits

Andrej Karpathy for the original proposal of the char-rnn via the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)