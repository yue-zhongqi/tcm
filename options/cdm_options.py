class CDMOptions():
    # def __init__(self):
    #    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    @staticmethod
    def initialize(parser):
        parser.add_argument('--s_dset', type=str, default="", help='source data txt file')
        parser.add_argument('--t_dset', type=str, default="", help='target data txt file')
        parser.add_argument('--s_name', type=str, default="A", help='source domain name')
        parser.add_argument('--t_name', type=str, default="D", help='target domain name')
        parser.add_argument('--dataset_name', type=str, default="office", help='dataset name')
        parser.add_argument('--cdm_path', type=str, default="/data2/xxxx/Model/dda/cdm", help='location of cross-domain mappings')
        parser.add_argument('--visualize', action="store_true", help="Turn on visualization mode")
        parser.add_argument('--all_experts', action="store_true", help="Use all experts")
        return parser

    @staticmethod
    def process_opt_str(opt, opt_str):
        return opt_str