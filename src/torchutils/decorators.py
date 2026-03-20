from typing import Callable, TypeVar, Generic, Optional
import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)
V = TypeVar("V", int, float, complex, bool)
class auxloss(property, Generic[M, V]):
    def __init__(
        self,
        fget: Callable[[M], torch.Tensor],
        fcollect: Optional[Callable[[M, Callable[[torch.Tensor], torch.Tensor], Optional[torch.dtype], Optional[torch.device]], torch.Tensor]] = None,
        fset: Optional[Callable[[M, torch.Tensor], None]] = None,
        freset: Optional[Callable[[M, V], None]] = None,
        doc: Optional[str] = None,
    ) -> None:
        super().__init__(fget, fset, None, doc)
        self._fcollect = fcollect
        self._freset = freset

    def setter(self, fset: Callable[[M, torch.Tensor], None]):
        # property.setter()는 "새 property"를 반환하므로, auxloss로 재구성
        p = super().setter(fset)  # type: ignore[misc]
        return auxloss(p.fget, self._fcollect, p.fset, self._freset, p.__doc__)

    def collector(self, fcollect):
        return auxloss(self.fget, fcollect, self.fset, self._freset, self.__doc__)

    def resetter(self, freset):
        return auxloss(self.fget, self._fcollect, self.fset, freset, self.__doc__)

    def reset(self, instance: M, value: V):
        if self._freset is None:
            raise AttributeError("No resetter defined")
        self._freset(instance, value)

    def collect(
        self,
        instance: M,
        aggregate: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._fcollect is None:
            raise AttributeError("No collector defined")
        return self._fcollect(instance, aggregate, dtype, device)