from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics
import sacrebleu


@register_criterion("dummy_criterion")
class CustomCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True):
        # Use the loss provided by the model
        outputs = model(**sample["net_input"])
        loss = outputs["loss"]
        nsentences = (sample["target"].size(0))
        logging_output = {
            "loss": outputs["loss"],
            "nsentences": nsentences,
            }

        return loss, nsentences, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        num_workers = len(logging_outputs)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)

        # BLEU
        hyps = []
        refs = []
        for log in logging_outputs:
            hyps.extend(log.get("hyps", []))
            refs.extend(log.get("refs", []))
        if len(hyps) > 0:
            assert len(hyps) == len(refs), "hyps and refs must have the same length"
            target_lang = logging_outputs[0].get("lang", None)
            assert target_lang is not None, "target_lang must be provided"
            if target_lang is not None and target_lang == "ja":
                bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="ja-mecab")
            elif target_lang is not None and target_lang == "zh":
                bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
            else:
                bleu = sacrebleu.corpus_bleu(hyps, [refs])
            bleu_sum = bleu.score
            metrics.log_scalar("bleu", bleu_sum, round=3)
            
        nsentences= sum(log.get("nsentences", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / num_workers, nsentences, round=3)
        
        

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        When loggging outputs are scalars and can be summed, set this to True to use _fast_stat_sync_sum.
        If the logging outputs are complex types, set this to False.
        """
        return False
