"""把 QLabs 的访问串行化 —— 修复多线程并发导致的段错误和"回包被吃掉"

## 背景（读 qvl 源码 + 实测得出）

`QuanserInteractiveLabs` 内部封着**一条 TCP 连接** —— `quanser.communications.Stream`，
是 Quanser 的 C/C++ 扩展 —— 外加一个**共享接收缓冲区**。它的 `wait_for_container`
设计上是**破坏性读取**，SDK 文档原话：

    Containers that do not match the class, actor number, and function number
    are discarded.

于是两个线程并发用同一个 `qlabs` 实例时：

* **Python 层**：互相吃掉对方的回包 → 谁先读谁拿走 → 另一个等到 5 秒超时 → 调用静默失败
* **C 层**：并发调用非线程安全的 `Stream.send_byte_array()` / `.receive()` → **段错误**

所以准确的说法不是"Python 线程不安全"，而是**两个线程共享了一个不可共享的连接对象**。
线程本身没问题，共享的东西有问题。

## 做法

给每个 QLabs actor 类挂一把全局 `RLock`，包住它所有公开方法，
使"一次请求-应答"整体串行：C 层不会被并发调用，缓冲区也不会被抢读。
用 `RLock` 是因为方法之间会互相调用（`spawn_id` 内部再调 `_is_actor_number_valid` 等），
可重入才不会自己把自己锁死。

## 代价几乎为零

`QLabsCharacter.move_to` 的 docstring 明确写了：

    This makes the method a blocking operation, but only until the command is
    received. The time for the actor to traverse to the destination is always
    non-blocking.

**行人走路不占锁** —— 锁只在"发指令 + 收回包"的几十毫秒里被持有，
行人走多久都不影响无人机控制。

## 用法

在使用任何 qvl 对象之前 `import Module.QLabsLock` 即可（幂等，重复 import 无副作用）。
只需要在**会开线程**的程序里 import；单线程程序加了也无害。
"""
import threading

from qvl.actor import QLabsActor

_LOCK = threading.RLock()
_WRAPPED = set()


def lock():
    """需要包住一段"含多次 QLabs 调用"的临界区时用它：`with QLabsLock.lock():`"""
    return _LOCK


def _wrap_method(fn, qualname):
    def wrapper(self, *args, **kwargs):
        with _LOCK:
            return fn(self, *args, **kwargs)
    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    wrapper.__qualname__ = qualname
    return wrapper


def _wrap_class(cls):
    """把 cls 自己的公开方法全部包上锁（不是继承来的，避免重复包装）"""
    if cls in _WRAPPED:
        return
    _WRAPPED.add(cls)
    for name, attr in list(vars(cls).items()):
        if name.startswith("_"):            # 跳过 __init__ / _is_actor_number_valid 等
            continue
        if isinstance(attr, (staticmethod, classmethod, property)):
            continue
        if not callable(attr):
            continue
        setattr(cls, name, _wrap_method(attr, f"{cls.__name__}.{name}"))


def _wrap_tree(cls):
    _wrap_class(cls)
    for sub in cls.__subclasses__():
        _wrap_tree(sub)


# ---- 1) 包装当前已经定义好的所有 actor 类 ----
_wrap_tree(QLabsActor)

# 非 actor 但同样走 qlabs 连接的两个类
try:
    from qvl.environment_outdoors import QLabsEnvironmentOutdoors
    _wrap_class(QLabsEnvironmentOutdoors)
except Exception:
    pass
try:
    from qvl.system import QLabsSystem
    _wrap_class(QLabsSystem)
except Exception:
    pass


# ---- 2) 之后新定义的 QLabsActor 子类也自动包装 ----
_orig_init_subclass = QLabsActor.__init_subclass__


def _init_subclass_hook(cls, **kwargs):
    _wrap_class(cls)


QLabsActor.__init_subclass__ = classmethod(_init_subclass_hook)


def report():
    """自检：现在有哪些类/方法被包上了"""
    n_cls = len(_WRAPPED)
    n_m = sum(1 for c in _WRAPPED for k in vars(c) if getattr(vars(c)[k], "__qualname__", "").startswith(c.__name__ + "."))
    print(f"QLabsLock：已包装 {n_cls} 个类、{n_m} 个方法")
    return n_cls, n_m


if __name__ == "__main__":
    report()
