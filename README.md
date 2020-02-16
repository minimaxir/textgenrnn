# textgenrnn

![dank text](/docs/textgenrnn_console.gif)

Easily train your own text-generating neural network of any size and complexity on any text dataset with a few lines of code, or quickly train on a text using a pretrained model.

textgenrnn is a Python 3 module on top of [Keras](https://github.com/fchollet/keras)/[TensorFlow](https://www.tensorflow.org) for creating [char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)s, with many cool features:

* A modern neural network architecture which utilizes new techniques as attention-weighting and skip-embedding to accelerate training and improve model quality.
* Train on and generate text at either the character-level or word-level.
* Configure RNN size, the number of RNN layers, and whether to use bidirectional RNNs.
* Train on any generic input text file, including large files.
* Train models on a GPU and then use them to generate text with a CPU.
* Utilize a powerful CuDNN implementation of RNNs when trained on the GPU, which massively speeds up training time as opposed to typical LSTM implementations.
* Train the model using contextual labels, allowing it to learn faster and produce better results in some cases.

You can play with textgenrnn and train any text file with a GPU *for free* in this [Colaboratory Notebook](https://drive.google.com/file/d/1mMKGnVxirJnqDViH7BDJxFqWrsXlPSoK/view?usp=sharing)! Read [this blog post](http://minimaxir.com/2018/05/text-neural-networks/) or [watch this video](https://www.youtube.com/watch?v=RW7mP6BfZuY) for more information!

## Examples

```python
from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.generate()
```

```text
[Spoiler] Anyone else find this post and their person that was a little more than I really like the Star Wars in the fire or health and posting a personal house of the 2016 Letter for the game in a report of my backyard.
```

The included model can easily be trained on new texts, and can generate appropriate text *even after a single pass of the input data*.

```python
textgen.train_from_file('hacker_news_2000.txt', num_epochs=1)
textgen.generate()
```

```text
Project State Project Firefox
```

The model weights are relatively small (2 MB on disk), and they can easily be saved and loaded into a new textgenrnn instance. As a result, you can play with models which have been trained on hundreds of passes through the data. (in fact, textgenrnn learns *so well* that you have to increase the temperature significantly for creative output!)

```python
textgen_2 = textgenrnn('/weights/hacker_news.hdf5')
textgen_2.generate(3, temperature=1.0)
```

```text
Why we got money “regular alter”

Urburg to Firefox acquires Nelf Multi Shamn

Kubernetes by Google’s Bern
```

You can also train a new model, with support for word level embeddings and bidirectional RNN layers by adding `new_model=True` to any train function.

## Interactive Mode

It's also possible to get involved in how the output unfolds, step by step. Interactive mode will suggest you the *top N* options for the next char/word, and allows you to pick one.  
  
When running textgenrnn in the terminal, pass `interactive=True` and `top=N` to `generate`. N defaults to 3.

```python
from textgenrnn import textgenrnn

textgen = textgenrnn()
textgen.generate(interactive=True, top_n=5)
```

![word_level_demo](/docs/textgenrnn_interactive.gif)
  
This can add a *human touch* to the output; it feels like you're the writer! ([reference](https://fivethirtyeight.com/features/some-like-it-bot/))
  
## Usage

textgenrnn can be installed [from pypi](https://pypi.python.org/pypi/textgenrnn) via `pip`:

```sh
pip3 install textgenrnn
```

For the latest textgenrnn, *you must have a minimum TensorFlow version of 2.1.0*.

You can view a demo of common features and model configuration options in [this Jupyter Notebook](/docs/textgenrnn-demo.ipynb).

`/datasets` contains example datasets using Hacker News/Reddit data for training textgenrnn.

`/weights` contains further-pretrained models on the aforementioned datasets which can be loaded into textgenrnn.

`/outputs` contains examples of text generated from the above pretrained models.

## Neural Network Architecture and Implementation

textgenrnn is based off of the [char-rnn](https://github.com/karpathy/char-rnn) project by [Andrej Karpathy](https://twitter.com/karpathy) with a few modern optimizations, such as the ability to work with very small text sequences.

![default model](/docs/default_model.png)

The included pretrained-model follows a [neural network architecture](https://github.com/bfelbo/DeepMoji/blob/master/deepmoji/model_def.py) inspired by [DeepMoji](https://github.com/bfelbo/DeepMoji). For the default model, textgenrnn takes in an input of up to 40 characters, converts each character to a 100-D character embedding vector, and feeds those into a 128-cell long-short-term-memory (LSTM) recurrent layer. Those outputs are then fed into *another* 128-cell LSTM. All three layers are then fed into an Attention layer to weight the most important temporal features and average them together (and since the embeddings + 1st LSTM are skip-connected into the attention layer, the model updates can backpropagate to them more easily and prevent vanishing gradients). That output is mapped to probabilities for up to [394 different characters](/textgenrnn/textgenrnn_vocab.json) that they are the next character in the sequence, including uppercase characters, lowercase, punctuation, and emoji. (if training a new model on a new dataset, all of the numeric parameters above can be configured)

![context model](/docs/context_model.png)

Alternatively, if context labels are provided with each text document, the model can be trained in a contextual mode, where the model learns the text *given the context* so the recurrent layers learn the *decontextualized* language. The text-only path can piggy-back off the decontextualized layers; in all, this results in much faster training and better quantitative and qualitative model performance than just training the model gien the text alone.

The model weights included with the package are trained on hundreds of thousands of text documents from Reddit submissions ([via BigQuery](http://minimaxir.com/2015/10/reddit-bigquery/)), from a very *diverse* variety of subreddits. The network was also trained using the decontextual approach noted above in order to both improve training performance and mitigate authorial bias.

When fine-tuning the model on a new dataset of texts using textgenrnn, all layers are retrained. However, since the original pretrained network has a much more robust "knowledge" initially, the new textgenrnn trains faster and more accurately in the end, and can potentially learn new relationships not present in the original dataset (e.g. the [pretrained character embeddings](http://minimaxir.com/2017/04/char-embeddings/) include the context for the character for all possible types of modern internet grammar).

Additionally, the retraining is done with a momentum-based optimizer and a linearly decaying learning rate, both of which prevent exploding gradients and makes it much less likely that the model diverges after training for a long time.

## Notes

* **You will not get quality generated text 100% of the time**, even with a heavily-trained neural network. That's the primary reason viral [blog posts](http://aiweirdness.com/post/170685749687/candy-heart-messages-written-by-a-neural-network)/[Twitter tweets](https://twitter.com/botnikstudios/status/955870327652970496) utilizing NN text generation often generate lots of texts and curate/edit the best ones afterward.

* **Results will vary greatly between datasets**. Because the pretrained neural network is relatively small, it cannot store as much data as RNNs typically flaunted in blog posts. For best results, use a dataset with at least 2,000-5,000 documents. If a dataset is smaller, you'll need to train it for longer by setting `num_epochs` higher when calling a training method and/or training a new model from scratch. Even then, there is currently no good heuristic for determining a "good" model.

* A GPU is not required to retrain textgenrnn, but it will take much longer to train on a CPU. If you do use a GPU, I recommend increasing the `batch_size` parameter for better hardware utilization.

## Future Plans for textgenrnn

* More formal documentation

* A web-based implementation using tensorflow.js (works especially well due to the network's small size)

* A way to visualize the attention-layer outputs to see how the network "learns."

* A mode to allow the model architecture to be used for chatbot conversations (may be released as a separate project)

* More depth toward context (positional context + allowing multiple context labels)

* A larger pretrained network which can accommodate longer character sequences and a more indepth understanding of language, creating better generated sentences.

* Hierarchical softmax activation for word-level models (once Keras has good support for it).

* FP16 for superfast training on Volta/TPUs (once Keras has good support for it).

## Articles/Projects using textgenrnn

### Articles

* Lifehacker: [How to Train Your Own Neural Network](https://lifehacker.com/we-trained-an-ai-to-generate-lifehacker-headlines-1826616918) by Beth Skwarecki
* New York Times: [Let Our Algorithm Choose Your Halloween Costume](https://www.nytimes.com/interactive/2018/10/26/opinion/halloween-spooky-costumes-machine-learning-generator.html) by Janelle Shane
* CNN Business: [This quirky experiment highlights AI's biggest challenges](https://www.cnn.com/2018/11/09/tech/janelle-shane-ai/index.html) by Rachel Metz

### Projects

* [Tweet Generator](https://github.com/minimaxir/tweet-generator) — Train a neural network optimized for generating tweets based off of any number of Twitter users
* [Hacker News Simulator](https://twitter.com/hackernews_nn) — Twitter bot trained on 300,000+ Hacker News submissions using textgenrnn.
* [SubredditRNN](https://www.reddit.com/r/subredditnn) — Reddit Subreddit where all submitted content is from textgenrnn bots.
* [Human-AI Collaborated Pizzas](https://howtogeneratealmostanything.com/food/2018/08/30/episode2.html) — Pizza recepies generated with textgenrnn and made in real life.
* [Board Game Titles](https://boardgamegeek.com/thread/2105706/i-trained-neural-network-17000-game-titles-bgg)
* [Video Game Discussion Forum Titles](https://www.resetera.com/threads/i-trained-an-ai-on-tens-of-thousands-of-resetera-post-titles-and-discovered-how-the-world-ends.82679/)
* [A.I Created Cakes](https://www.cupcaikes.com/index.html)
* [AI Created Cookies](http://aiweirdness.com/post/180892528177/aw-yeah-its-time-for-cookies-with-neural-networks)
* [AI Generated Songs](http://aiweirdness.com/post/180654319147/how-to-begin-a-song)

### Tweets

* [BuzzFeed YouTube Videos](https://twitter.com/minimaxir/status/1064604986951163905)
* [AWS Services](https://twitter.com/jamesoff/status/1073647847130742787)
* [Recipes + D&D Spells + Heavy Metal Names](https://twitter.com/ThomasClaburn/status/1049069940571955201)
* [RPG Adventure Names](https://twitter.com/400goblins/status/1036794962740953088)
* [The Onion + Cosmopolitan](https://twitter.com/BBCPARLlAMENT/status/1014834653113585664)
* [Google Conference Room Names](https://twitter.com/tensafefrogs/status/1009912151060951045)
* [Sith Lords](https://twitter.com/JanelleCShane/status/1002573232103305216)

## Maintainer/Creator

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## Credits

Andrej Karpathy for the original proposal of the char-rnn via the blog post [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

[Daniel Grijalva](https://github.com/Juanets) for [contributing](https://github.com/minimaxir/textgenrnn/pull/52) an interactive mode.

## License

MIT

Attention-layer code used from [DeepMoji](https://github.com/bfelbo/DeepMoji) (MIT Licensed)
