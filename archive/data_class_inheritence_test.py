from dataclasses import dataclass

@dataclass(frozen=True)
class MyBaseDataClass:
    a: int
    b: int

    def __post_init__(self):
        self.a_plus_b = self.a + self.b

@dataclass(frozen=True)
class MySubDataClass(MyBaseDataClass):
    c: int
