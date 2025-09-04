import argparse
from ml_util.docux_logger import give_logger, configure_logger
from ml_util.sentence_transformer_interface import SentenceTransformerTrainerHolder
from typing import Tuple, Dict, List, Iterable

logger = give_logger('supConLearn')

class RawCPT:
    skip_fields = ('Concept Id', 'Current Descriptor', 'Effective Date', 'Test Name', 'Lab Name', 'Manufacturer Name')
    '''
    Concept Id	CPT Code	Long	Medium	Short	Consumer	Spanish Consumer	Current Descriptor Effective Date	
    Test Name	Lab Name	Manufacturer Name
    '''
    display_fields = ('Long', 'Consumer')
    def __init__(self,
                 code_file: str):
        self.by_cpt: Dict[str, Tuple[str]] = {}
        self.header_inds = []
        self.field_names = []
        cpt_ind = None
        with open(code_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                if len(self.header_inds) < 1:
                    if line.startswith("Concept Id"):
                        fields = line.strip().split("\t")
                        for ind, field in enumerate(fields):
                            if field not in self.skip_fields:
                                self.header_inds.append(ind)
                                self.field_names.append(field)
                    cpt_ind = self.field_names.index('CPT Code')
                else:
                    raw: List[str] = line.strip().split("\t")
                    use_values = tuple([raw[i] for i in self.header_inds])
                    cpt = use_values[cpt_ind]
                    self.by_cpt[cpt] = use_values

        self.value_for_cpt_field = lambda cpt, field: (
            self.by_cpt[
                cpt
            ][
                self.field_names.index(field)
            ])

    def give_variants_for_cpt(self,
                              cpt_code: str) -> Tuple:
        return tuple([self.value_for_cpt_field(cpt_code, f)
                      for f in  self.display_fields])

    def give_all_variants(self) -> Dict[str, List[str]]:
        out = \
            {cpt:
                sorted(list(set(
                    [fields[
                         self.field_names.index(n)
                     ]
                     for n in self.display_fields]
                )))
                for cpt, fields in self.by_cpt.items()}

        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_code_file', type='str', default='Consolidated_Code_List.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()

    args.do_train = True
    args.do_eval = True
    args.do_eval = False

    cpt_table = RawCPT(args.cpt_code_file)

    # Derive datasets from cpt_table...

    trainer = SentenceTransformerTrainerHolder.create(args,
                                                      args.model_name,
                                                      )






if __name__ == '__main__':
    main()