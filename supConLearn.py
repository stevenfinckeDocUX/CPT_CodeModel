import argparse
from datasets import DatasetDict
from transformers import Trainer
from ml_util.random_utils import set_seed
from ml_util.classes import ClassMember, ClassInventory
from ml_util.docux_logger import give_logger
from ml_util.supervised_contrastive import SentenceTransformerSupConTrainer
from ml_util.supervised_contrastive import SentenceTransformerSupConTrainer
from ml_util.batch_all import get_BatchAll_train_dev_test_dict, BatchCache
from ml_util.triplet import get_Triplet_train_dev_test_dict, SentenceTransformerTripletTrainer, SentenceTransformerAllBatchTripletTrainer
from ml_util.sentence_transformer_interface import SentenceTransformerCustomTrainer
from typing import Tuple, Dict, List, Type

logger = give_logger('supConLearn')

class RawCPT:
    skip_fields = ('Concept Id', 'Current Descriptor Effective Date', 'Test Name', 'Lab Name', 'Manufacturer Name',
                   'Spanish Consumer')
    '''
    Concept Id	CPT Code	Long	Medium	Short	Consumer	Spanish Consumer	Current Descriptor Effective Date	
    Test Name	Lab Name	Manufacturer Name
    '''
    display_fields: List[str] = ['Long', 'Consumer']
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
        class_inventory = ClassInventory(name='CPT Inventory')

        for cpt, fields in sorted(self.by_cpt.items()):
            ready_fields = sorted(list(set(
                [
                    fields[
                        self.field_names.index(n)
                    ]
                    for n in self.display_fields]
            )))
            if len(ready_fields) >= min_form_count_per_class:
                class_inventory.add_member(cpt, tuple(ready_fields))

        return class_inventory


supported_loss = ('SupCon', 'Triplet', 'BATriplet', 'BShATriplet', 'VBATriplet')


def get_train_dev_test_dict(cpt_inventory: ClassInventory,
                            args: argparse.PARSER,
                            batch_cache: BatchCache) -> DatasetDict:
    loc_args = (cpt_inventory, args.part_train, args.part_test)
    loc_kwargs = {'shuffle': args.shuffle_data, 'seed': args.seed}

    if args.loss in ('SupCon'):
        return get_BatchAll_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss == 'Triplet':
        return get_Triplet_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss in ('BATriplet', 'BShATriplet', 'VBATriplet'):
        return get_BatchAll_train_dev_test_dict(*loc_args,
                                                batch_cache=batch_cache,
                                                **loc_kwargs,
                                                # label_field_name='label', input_field_name='sentence'
                                                )
    else:
        raise NotImplementedError


trainer_class_map: Dict[str, Type] = \
    {'SupCon': SentenceTransformerSupConTrainer,
     'Triplet': SentenceTransformerTripletTrainer,
     'BATriplet': SentenceTransformerAllBatchTripletTrainer,
     'BShATriplet': SentenceTransformerAllBatchTripletTrainer,
     'VBATriplet': SentenceTransformerAllBatchTripletTrainer,
     }


def get_trainer(args: argparse.PARSER,
                class_inventory: ClassInventory = None) -> SentenceTransformerCustomTrainer:
    batch_cache = BatchCache(args.per_device_train_batch_size) if args.hard_batching else None
    dataset_dict = get_train_dev_test_dict(class_inventory, args, batch_cache)
    loc_kwargs = {'top_args': args,
                  'model_name': args.model_name,
                  'train_dataset': dataset_dict['train'],
                  'eval_dataset': dataset_dict['valid'],
                  'loss_name': args.loss,
                  'class_inventory': class_inventory,
                  'batch_cache': batch_cache,
                  }
    trainer_class = trainer_class_map[args.loss]

    return trainer_class(**loc_kwargs)

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
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--output_hidden_states', action='store_true')
    parser.add_argument('--hard_batching', action='store_true',
                        help="Keep model output embeddings for subsequent hard batching.")
    args = parser.parse_args()
    # configure_logger(logger, args.log_file, level=args.logging_level)

    set_seed(args.seed)

    # For the trainer
    args.do_train = True
    args.do_eval = True
    args.do_predict = False
    args.eval_strategy = 'epoch'

    raw_cpt_table = RawCPT(args.cpt_code_file,
                           required_fields=args.required_fields,
                           required_init_strings=args.init_cpt_filters)
    print(f"raw cpt cnt: {len(raw_cpt_table.by_cpt)}")
    cpt_inventory = raw_cpt_table.give_inventory(min_form_count_per_class=len(args.required_fields))

    # dataset_dict = get_train_dev_test_dict(cpt_inventory, args)
    trainer = get_trainer(args, cpt_inventory)
    init_metrics = trainer.evaluate()
    print(f"init_metrics: {init_metrics}")

    trainer.train()


if __name__ == '__main__':
    main()
