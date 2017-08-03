# textgenrnn

textgenrnn is a Python module which can easily generate text using a pretrained recurrent neural network:

```python
import textgenrnn

model = textgenrnn()
model.generate()
```

And can easily be trained on new texts:

```python
model.train_from_text('buzzfeed_headlines.csv')
model.generate()
```

The model is extremely small (under 1MB on disk), and the weights can easily be saved and loaded into any application.

```python
model.save('buzzfeed_model.hdf5')

model_2 = textgenrnn('buzzfeed_model.hdf5')
```

## Usage


## Neural Network Architecture

textgenrnn is based off of the char-rnn by Andrej Karpathy with a few optimizations, including the ability to generate smaller text sequences, and text without requiring a seed text.

The provided pretrained network is trained on hundreds of thousands of text documents from Reddit submissions (via BigQuery) and Facebook Pages (via my Facebook Page Post Scraper), from a large variety of subreddits/Pages. The network was also trained in such a way that the `rnn` is decontextualized in order to both improve training accuracy and mitigate authorial/sampling bias.

When training on a new dataset, the Embeddings layer is frozen while the `rnn` and `output` layers are retrained. However, since the source pretrained network has a much wider breath of knowledge already within, the new textgenrnn trains faster and more accurately in the end, and can learn a few new things potentially not present in the original dataset. Additionally, the retraining is done with the Nadam optimizer and a linearly decaying learning rate, the combination of which prevents exploding gradients and assures learning.

## Notes

* A GPU is not required to retrain textgenrnn, but it may take awhile to train on a CPU. (a couple hours for a dataset with a couple thounsand documents). If you do use a GPU, I recommend increasing the `batch_size` for better utilization, and increasing `num_epochs` to compensate.

* textgenrnn is less effective when training on/predicting longer sequences (> 200 characters). Likewise textgenrnn is less effective when training on/predicting texts with very disparate styles. If a source dataset has *both*, it may lead to unexpected results.

## Future Plans for textgenrnn

* A web-based implementation using Keras.js (works especially well due to the network's small size)

* A way to visualize the outputs of the RNN's neurons.

* A larger pretrained network which can accommodate longer character sequences and a more indepth understanding of language. This may be released as a commercial product, if any venture capitalists have loose change.

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## Credits

Andrej Karpathy for the original implementation of char-rnn via the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)