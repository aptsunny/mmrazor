from abc import ABC

class SearchableBackboneMixin(ABC):

    def load_backbone(self, fix_subnet):
        if fix_subnet:
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self, fix_subnet)
