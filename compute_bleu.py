import re
import sys
import unicodedata

from absl import app as absl_app
from absl import flags,logging
import tensorflow as tf

from utils import metrics
from utils.flags import core as flags_core

class UnicodeRegex(object):
    """Ad-hoc hack to recognize all puntuation and symbols."""

    def __init__(self):
        puntuation=self.property_chars('P')
        self.nondigit_punct_re=re.compile(r'([^\d])([)'+puntuation+r'])')
        self.punct_nondigit_re=re.compile(r'(['+puntuation+r'])([^\d])')
        self.symbol_re=re.compile('(['+self.property_chars('S'))+'])'

    def property_chars(self,prefix):
        return "".join(chr(x) for x in range(sys.maxunicode)
                       if unicodedata.category(chr(x)).startswith(prefix))

uregex=UnicodeRegex()

def bleu_tokenize(string):
    string=uregex.nondigit_punct_re.sub(r'\1 \2 ',string)
    string=uregex.punct_nondigit_re.sub(r' \1 \2',string)
    string=uregex.symbol_re.sub(r' \1',string)
    return string.split()

def bleu_wrapper(ref_filename,hyp_filename,case_sensitive=False):
    """Compute BLEU for two files (reference and hypothesis translation)."""
    ref_lines=tf.io.gfile.GFile(ref_filename).read().strip().splitlines()
    hyp_lines=tf.io.gfile.GFile(hyp_filename).read().strip().splitlines()

    if len(ref_lines)!=len(hyp_lines):
        raise ValueError("Reference and translation files have different number of "
                         "lines. If training only a few steps (100-200), the "
                         "translation may be empty.")

    if not case_sensitive:
        ref_lines=[x.lower() for x in ref_lines]
        hyp_lines=[x.lower() for x in hyp_lines]
    ref_tokens=[bleu_tokenize(x) for x in ref_lines]
    hyp_tokens=[bleu_tokenize(x) for x in hyp_lines]
    return metrics.compute_bleu(ref_tokens,hyp_tokens)*100

def main(unused_argv):
  if FLAGS.bleu_variant in ("both", "uncased"):
    score = bleu_wrapper(FLAGS.reference, FLAGS.translation, False)
    logging.info("Case-insensitive results: %f" % score)

  if FLAGS.bleu_variant in ("both", "cased"):
    score = bleu_wrapper(FLAGS.reference, FLAGS.translation, True)
    logging.info("Case-sensitive results: %f" % score)

def define_compute_bleu_flags():
  """Add flags for computing BLEU score."""
  flags.DEFINE_string(
      name="translation", default=None,
      help=flags_core.help_wrap("File containing translated text."))
  flags.mark_flag_as_required("translation")

  flags.DEFINE_string(
      name="reference", default=None,
      help=flags_core.help_wrap("File containing reference translation."))
  flags.mark_flag_as_required("reference")

  flags.DEFINE_enum(
      name="bleu_variant", short_name="bv", default="both",
      enum_values=["both", "uncased", "cased"], case_sensitive=False,
      help=flags_core.help_wrap(
          "Specify one or more BLEU variants to calculate. Variants: \"cased\""
          ", \"uncased\", or \"both\"."))


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  define_compute_bleu_flags()
  FLAGS = flags.FLAGS
  absl_app.run(main)