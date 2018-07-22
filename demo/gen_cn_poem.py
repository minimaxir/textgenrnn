from textgenrnn import textgenrnn

textgen = textgenrnn(
  name="poem",
  weights_path='./poem/poem_weights.hdf5',
  config_path='./poem/poem_config.json',
  vocab_path='./poem/poem_vocab.json'
)
textgen.generate(20, temperature=1.0)