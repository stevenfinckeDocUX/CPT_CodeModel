import argparse
from dataclasses import dataclass
from collections import UserList
from datasets import DatasetDict
from transformers import Trainer
from ml_util.random_utils import set_seed
from ml_util.classes import ClassMember, ClassInventory
from ml_util.docux_logger import give_logger, configure_logger
from ml_util.supervised_contrastive import get_BatchAll_train_dev_test_dict
from ml_util.triplet import get_Triplet_train_dev_test_dict
from ml_util.sentence_transformer_interface import SentenceTransformerCustomTrainer
from typing import Tuple, Dict, List, Iterable

logger = give_logger('supConLearn')

class RawCPT:
    skip_fields = ('Concept Id', 'Current Descriptor Effective Date', 'Test Name', 'Lab Name', 'Manufacturer Name',
                   'Spanish Consumer')
    '''
    Concept Id	CPT Code	Long	Medium	Short	Consumer	Spanish Consumer	Current Descriptor Effective Date	
    Test Name	Lab Name	Manufacturer Name
    '''
    display_fields = ('Long', 'Consumer')
    def __init__(self,
                 code_file: str,
                 *,
                 required_init_strings: List[str] = None,
                 required_fields: List[str] = None
                 ):
        self.by_cpt: Dict[str, Tuple[str]] = {}
        self.header_inds = []
        self.field_names: List[str] = []

        cpt_is_usable = \
            lambda cpt: (required_init_strings is None
                         or sum([int(cpt.startswith(init_s)) for init_s in required_init_strings]) > 0)

        cpt_ind = None
        required_inds = None
        with open(code_file, "r", encoding='utf-8') as in_H:
            for line in in_H:
                line = line.strip()
                if len(self.header_inds) < 1:
                    if line.startswith("Concept Id"):
                        fields = line.strip().split("\t")
                        for ind, field in enumerate(fields):
                            if field not in self.skip_fields:
                                self.header_inds.append(ind)
                                self.field_names.append(field)
                        cpt_ind = self.field_names.index('CPT Code')
                        if required_fields:
                            required_inds = [self.field_names.index(n) for n in required_fields]
                else:
                    line = line.strip()
                    raw: List[str] = line.split("\t")
                    use_values = tuple([raw[i] if i < len(raw) else ''
                                        for i in self.header_inds])
                    if required_fields and min([len(use_values[ind]) for ind in required_inds]) < 1:
                        logger.info(f"skip input line because it lacks required values: {line.strip()}")
                        continue
                    cpt = use_values[cpt_ind]
                    if cpt_is_usable(cpt):
                        self.by_cpt[cpt] = use_values

        self.value_for_cpt_field = lambda cpt, field: (
            self.by_cpt[
                cpt
            ][
                self.field_names.index(field)
            ])
        pass

    def give_variants_for_cpt(self,
                              cpt_code: str) -> Tuple:
        return tuple([self.value_for_cpt_field(cpt_code, f)
                      for f in  self.display_fields])

    def give_inventory(self,
                       min_form_count_per_class: int) -> ClassInventory:
        ready: List[ClassMember] = []
        for cpt, fields in sorted(self.by_cpt.items()):
            ready_fields = tuple(sorted(list(set(
                [
                    fields[
                        self.field_names.index(n)
                    ]
                    for n in self.display_fields]
            ))))
            if len(ready_fields) >= min_form_count_per_class:
                ready.append(ClassMember(cpt, ready_fields))

        return ClassInventory(ready, name='CPT Inventory')

supported_loss = ('SupCon', 'Triplet', 'BATriplet', 'BShATriplet')
def get_train_dev_test_dict(gpt_inventory: ClassInventory, args: argparse.PARSER) -> DatasetDict:
    loc_args = (gpt_inventory, args.part_train, args.part_test)
    loc_kwargs = {'shuffle': args.shuffle_data, 'seed': args.seed}

    if args.loss in ('SupCon'):
        return get_BatchAll_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss == 'Triplet':
        return get_Triplet_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss in ('BATriplet', 'BShATriplet'):
        return get_BatchAll_train_dev_test_dict(*loc_args, **loc_kwargs,
                                                label_field_name='label', input_field_name='sentence')
    else:
        raise NotImplementedError

def give_trainer(args: argparse.PARSER, dataset_dict: DatasetDict) -> Trainer:
    loc_args = (args,
                args.model_name,
                dataset_dict['train'],
                dataset_dict['valid'],
                )
    loc_kwargs = {'loss_name': args.loss}
    return SentenceTransformerCustomTrainer(*loc_args, **loc_kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpt_code_file', type=str, default='Consolidated_Code_List.txt')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_name', type=str,
                        default="pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--overwrite_output_dir', action='store_true')
    parser.add_argument('--per_device_train_batch_size', type=int, default=128)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--part_train', type=float, default=0.9)
    parser.add_argument('--part_test', type=float, default=0.05)
    parser.add_argument('--shuffle_data', action='store_true')
    parser.add_argument('--init_cpt_filters', type=str, nargs='+',
                        help="Only use CPT codes which begin with one of these strings.")
    parser.add_argument('--loss_temperature', type=float, default=0.01)
    parser.add_argument('--evaluation_strategy', type=str, default='epoch')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--log_file', type=str, default="sup_con.log")
    parser.add_argument('--logging_level', type=str, default='DEBUG')
    parser.add_argument('--loss', type=str, default='SupCon', choices=supported_loss)
    parser.add_argument('--required_fields', type=str, nargs='+', default=['Long', 'Consumer'])
    parser.add_argument('--triplet_loss_margin', type=float, default=0.5)
    args = parser.parse_args()
    # configure_logger(logger, args.log_file, level=args.logging_level)

    set_seed(args.seed)

    # For the trainer
    args.do_train = True
    args.do_eval = True
    args.do_predict = False

    raw_cpt_table = RawCPT(args.cpt_code_file,
                           required_fields=args.required_fields,
                           required_init_strings=args.init_cpt_filters)
    print(f"raw cpt cnt: {len(raw_cpt_table.by_cpt)}")
    gpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=len(args.required_fields))

    dataset_dict = get_train_dev_test_dict(gpt_inventory, args)
    trainer = give_trainer(args, dataset_dict)

    trainer.train()


if __name__ == '__main__':
    main()
