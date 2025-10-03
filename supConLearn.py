import argparse
from datasets import DatasetDict
import numpy as np
import torch

from typing import Tuple, Dict, List, Type

from nltk.metrics.aline import similarity_matrix
from numpy.core.records import ndarray
from sentence_transformers import SentenceTransformer
from scipy.stats import pearsonr, spearmanr

from ml_util.faiss_interface import KMeansWeightedClusters, KMeansWeighted
from ml_util.random_utils import set_seed
from ml_util.classes import ClassInventory
from ml_util.docux_logger import give_logger
from ml_util.supervised_contrastive import SentenceTransformerSupConTrainer
from ml_util.batch_all import get_BatchAll_train_dev_test_dict, BatchCache, give_ranges_by_common, BatchAllDataset, \
    give_ranges_by_common
from ml_util.triplet import get_Triplet_train_dev_test_dict, SentenceTransformerTripletTrainer, \
    SentenceTransformerAllBatchTripletTrainer
from ml_util.sentence_transformer_interface import SentenceTransformerCustomTrainer, SentenceTransformerHolder

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
                            batch_cache: BatchCache,
                            ) -> DatasetDict:
    loc_args = ()
    loc_kwargs = {'class_inventory': cpt_inventory,
                  'part_train': args.part_train,
                  'part_test': args.part_test,
                  'shuffle': args.shuffle_data,
                  'seed': args.seed,
                  'hard_batching': args.hard_batching,
                  'train_batch_size': args.per_device_train_batch_size,
                  'test_batch_size': args.per_device_eval_batch_size,
                  }

    if args.loss in ('SupCon'):
        return get_BatchAll_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss == 'Triplet':
        return get_Triplet_train_dev_test_dict(*loc_args, **loc_kwargs)
    elif args.loss in ('BATriplet', 'BShATriplet', 'VBATriplet'):
        return get_BatchAll_train_dev_test_dict(*loc_args,
                                                train_batch_cache=batch_cache,
                                                **loc_kwargs,
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
    train_batch_cache = BatchCache(args.per_device_train_batch_size) if args.hard_batching else None
    eval_batch_cache = BatchCache(args.per_device_eval_batch_size)
    dataset_dict = get_train_dev_test_dict(class_inventory, args, train_batch_cache)
    loc_kwargs = {'top_args': args,
                  'model_name': args.model_name,
                  'train_dataset': dataset_dict['train'],
                  'eval_dataset': dataset_dict['valid'],
                  'loss_name': args.loss,
                  'class_inventory': class_inventory,
                  'train_batch_cache': train_batch_cache,
                  'eval_batch_cache': eval_batch_cache,
                  }
    trainer_class = trainer_class_map[args.loss]

    return trainer_class(**loc_kwargs)


class SnapShot:
    def __init__(self,
                 gpt_inventory: ClassInventory,
                 holder: SentenceTransformerHolder,
                 train_dataset: BatchAllDataset,
                 ):
        self.gpt_inventory = gpt_inventory
        self.strings, self.labels, self.string_inds = gpt_inventory.get_flat()
        self.space_size = self.labels.shape[0]
        vecs = holder.encode_no_grad(self.strings)
        self.km = KMeansWeighted(vecs,
                                 self.labels,
                                 np.array(give_ranges_by_common()),
                                 gpt_inventory,
                                 )

        self.embedding_sim_matrix = self.km.raw_similarity_matrix

        self.train_dataset = train_dataset
        self._train_mask: np.ndarray[bool] = None
        self._entry_similarity_matrix: np.ndarray[int] = None

    @property
    def train_inds(self) -> List[int]:
        return self.train_dataset.all_flat_inds

    @property
    def other_inds(self) -> List[int]:
        return [i for i in range(self.space_size) if i not in self.train_inds]

    @property
    def train_mask(self) -> np.ndarray[bool]:
        if self._train_mask is None:
            train_mask = np.zeros_like(self.labels, dtype=np.bool_)
            train_mask[self.train_dataset.all_flat_inds] = True
            train_mask = np.expand_dims(train_mask, axis=0).repeat(len(self.labels), axis=0)
            train_mask = train_mask * train_mask.T
            self._train_mask = train_mask

        return self._train_mask

    def _get_entry_similarity_matrix(self):
        by_row = np.repeat(np.expand_dims(self.labels, 0), self.labels.shape[0], axis=0)
        entry_similarity_matrix = self.gpt_inventory.label_similarity_matrix[by_row, by_row.T]
        self._entry_sim_values = np.unique(entry_similarity_matrix)
        # The max value is only for identity...
        self._top_val = self._entry_sim_values.max()
        self._entry_similarity_matrix = (
                entry_similarity_matrix +
                (torch.eye(entry_similarity_matrix.shape[0]) * (1 + self._entry_sim_values.max()))
        )

    @property
    def entry_similarity_matrix(self) -> np.ndarray[int]:
        if self._entry_similarity_matrix is None:
            self._get_entry_similarity_matrix()
        return self._entry_similarity_matrix

    @property
    def entry_sim_values(self) -> List[int]:
        if self._entry_sim_values is None:
            self._get_entry_similarity_matrix()
        return self._entry_sim_values

    @property
    def top_val(self) -> int:
        if self._top_val is None:
            self._get_entry_similarity_matrix()
        return self._top_val

    @property
    def sorted_sim_ranks(self) -> np.ndarray[int]:
        return self.entry_similarity_matrix.sort(axis=1, descending=True)[0]

    @property
    def cossim_ranks(self) -> np.ndarray[int]:
        # 1 for identity...
        return self.space_size - 1 - self.embedding_sim_matrix.argsort(axis=1).argsort(axis=1)

    def get_correlations(self,
                         *,
                         use_inds: List[int] = None):
        ind_prep = lambda m: m[use_inds] if use_inds is not None else m

        pearson = pearsonr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)
        spearman = spearmanr(ind_prep(self.entry_similarity_matrix), ind_prep(self.embedding_sim_matrix), axis=1)

        return pearson.correlation.mean(), spearman.correlation.mean()

    @property
    def top_sim_inds(self) -> torch.Tensor:
        return torch.argwhere(self.entry_similarity_matrix == self.top_val)

    def get_top_rank_analysis(self,
                              use_inds: List[int] = None):
        use_top_sim_inds = self.top_sim_inds if use_inds is None else self.top_sim_inds[use_inds]
        top_ranks = self.cossim_ranks[use_top_sim_inds.T[0], use_top_sim_inds.T[1]]

        denom = self.space_size if use_inds is None else len(use_inds)
        top = [f"{(top_ranks <= n).sum() / float(denom):.3f}" for n in range(1, 10)]

        return (f"top rank mean: {top_ranks.mean():.3f} ({top_ranks.std()}, {top_ranks.min()}-{top_ranks.max()})\t"
                f"top 10: {top}")

    def get_sim_errors(self) -> Dict:
        out = {}
        for split, loc_inds in (('train', self.train_inds), ('other', self.other_inds)):
            for_cossim = torch.stack(
                [es[rc] for es, rc in
                 zip(self.entry_similarity_matrix[loc_inds], self.cossim_ranks[loc_inds] - 1)])
            diffs = (self.sorted_sim_ranks[loc_inds] - for_cossim)
            # Identity comes first...
            diffs[:, 0] = 0
            diffs = diffs.clip(min=0, max=100)
            diffs = diffs.unique(return_counts=True)
            out[split] = {
                'diffs': diffs,
                'str': f"{diffs[1]}\n"
                       f"{self.get_correlations(use_inds=loc_inds)}\n"
                       f"{self.get_top_rank_analysis(use_inds=loc_inds)}"}

        return out

    def give_train_masked(self, src: np.ndarray[bool],
                          *,
                          in_mask: np.ndarray[bool] = None):
        if in_mask is None:
            in_mask = 1
        train_inds = np.argwhere(in_mask * self.train_mask).T
        other_inds = np.argwhere(in_mask * ~self.train_mask).T

        return src[train_inds[0], train_inds[1]], src[other_inds[0], other_inds[1]]


    def compare_to_prev(self,
                        prev: 'SnapShot',
                        ):
        assert np.array_equal(self.labels, prev.labels)

        prev_sim_error_d = prev.get_sim_errors()
        curr_sim_error_d = self.get_sim_errors()
        print(f"sim errors: {prev_sim_error_d['train']['diffs'][0]}"
              f"\nprev train: {prev_sim_error_d['train']['str']}\nprev other: {prev_sim_error_d['other']['str']}\n"
              f"\ncurr train: {curr_sim_error_d['train']['str']}\ncurr other: {curr_sim_error_d['other']['str']}")

        give_loc_stat_str = lambda loc: f"{np.mean(loc):.3f} ({np.std(loc):.3f}, {loc.min():.3f}-{loc.max():.3f})"

        give_stat_str = lambda src, in_mask: (
            " ".join([f"{n}\t{give_loc_stat_str(loc)}"
                      for n, loc in zip(('train', 'other'), self.give_train_masked(src, in_mask=in_mask))]))

        for sim in self.entry_sim_values:
            mask = self.entry_similarity_matrix.numpy() == sim
            print(f"sim: {sim} count: {mask.sum()}")
            for n, ranks in (('prev', prev.cossim_ranks), ('curr', self.cossim_ranks)):
                print(f"sim: {sim}\tranks {n}:\t{give_stat_str(ranks, mask)}")

            for n, all_sims in (('prev', prev.embedding_sim_matrix),
                                ('curr', self.embedding_sim_matrix),
                                ):
                print(f"sim: {sim}\tdiffs {n}:\t{give_stat_str(all_sims, mask)}")


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
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--fp16', action='store_true', help="Must be on GPU!")
    parser.add_argument("--torch_empty_cache_steps", type=int, default=1 ,
                        help="Needed, at least, when running of a MacBook with 24GB physical RAM (MPS).")
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

    trainer = get_trainer(args, cpt_inventory)
    init_metrics = trainer.evaluate()
    init_snapshot = SnapShot(cpt_inventory, trainer.holder, trainer.train_dataset)
    print(f"init_metrics: {init_metrics}")
    trainer.train()
    final_snpshot = SnapShot(cpt_inventory, trainer.holder, trainer.train_dataset)

    final_snpshot.compare_to_prev(init_snapshot)
    print(f"{give_ranges_by_common()}")


if __name__ == '__main__':
    main()
