import enum


class AllocStatus(enum.Enum):
    OK = 0
    LATER = 1
    NEVER = 2


class BlockManager:
    """Free-list allocator for paged blocks or state slots (upstream ``csrc/engine/block_manager.hpp``, no pinning)."""

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free = list(range(num_blocks - 1, -1, -1))

    def get_num_free_blocks(self) -> int:
        return len(self._free)

    def can_allocate(self, num_blocks: int) -> AllocStatus:
        if num_blocks > self.num_blocks:
            return AllocStatus.NEVER
        return AllocStatus.OK if num_blocks <= len(self._free) else AllocStatus.LATER

    def allocate(self, num_blocks: int) -> list[int]:
        if num_blocks > len(self._free):
            raise RuntimeError(f"cannot allocate {num_blocks} blocks, {len(self._free)} free")
        return [self._free.pop() for _ in range(num_blocks)]

    def allocate_single(self) -> int:
        return self.allocate(1)[0]

    def free_blocks(self, block_ids: list[int]) -> None:
        self._free.extend(reversed(block_ids))
