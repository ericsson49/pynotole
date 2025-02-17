from typing import Never


def fail(msg: str) -> Never:
    raise Exception(msg)


def fail_if(condition, msg: str):
    if condition:
        fail(msg)