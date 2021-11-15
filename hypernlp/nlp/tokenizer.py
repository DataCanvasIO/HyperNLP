import abc

from transformers import AutoTokenizer

from hypernlp.config import Config


class Tokenizer(object):

    def __init__(self, model_path, max_len):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path)
        if Config.framework == "pytorch":
            self.return_tensors = "pt"

        elif Config.framework == "tensorflow":
            self.return_tensors = "tf"

        else:
            raise ValueError("Unsupported framework: {}!".format(Config.framework))
        self.max_len = max_len

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def vocab_size(self):
        return self.tokenizer.vocab_size

    def mask_token_id(self):
        return self.tokenizer.mask_token_id


class TokenizerCLS(Tokenizer):

    def __init__(self, model_path, max_len):
        super(TokenizerCLS, self).__init__(model_path, max_len)

    def __call__(self, *args, **kwargs):

        assert len(args) == 1
        seq = args[0][0]

        if len(seq) >= 512:
            seq = seq[:512]

        return self.tokenizer(seq,
                              padding='max_length',  # Pad to max_length
                              truncation=True,  # Truncate to max_length
                              max_length=self.max_len,  # Set max_length
                              return_tensors=self.return_tensors)  # Return tf.Tensor objects


class TokenizerNSP(Tokenizer):

    def __init__(self, model_path, max_len):
        super(TokenizerNSP, self).__init__(model_path, max_len)

    def __call__(self, *args, **kwargs):

        seq1, seq2 = args[0][0], args[0][1]

        if len(seq1) >= 512:
            seq1 = seq1[:512]

        if len(seq2) >= 512:
            seq2 = seq2[:512]

        return self.tokenizer(seq1, seq2,
                              padding='max_length',  # Pad to max_length
                              truncation=True,  # Truncate to max_length
                              max_length=self.max_len,  # Set max_length
                              return_tensors=self.return_tensors)  # Return pt.Tensor objects


if __name__ == '__main__':
    from utils.string_utils import home_path, generate_model_name
    from hypernlp.dl_framework_adaptor.configs.config import bert_models_config

    nsp_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "cased")]["BASE_MODEL_PATH"], max_len=128)

    seq1 = 'Running periodic task ComputeManager._poll_unconfirmed_resizes run_periodic_tasks /usr/lib/python2.7/site-packages/oslo_service/periodic_task.py:215'
    seq2 = 'Running cmd (subprocess): /usr/bin/python2 -m oslo_concurrency.prlimit --as=1073741824 --cpu=2 -- env LC_ALL=C LANG=C qemu-img info /var/lib/nova/instances/3ad77c0d-f281-408d-8c08-ebd39514a014/disk execute /usr/lib/python2.7/site-packages/oslo_concurrency/processutils.py:344'

    line = ''
    for seq in seq1.split(' '):
        # if seq == 'task':
        #     line += '[MASK] '
        # else:
        line += seq + ' '
    line = line[:-1]
    print(line)
    d = [line, seq2]

    encoded_pair = nsp_tokenizer(d)

    print(nsp_tokenizer.vocab_size())

    print(encoded_pair['input_ids'],
                      encoded_pair['attention_mask'], encoded_pair['token_type_ids'])

