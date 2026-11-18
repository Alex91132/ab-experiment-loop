"""Планирование эксперимента: MDE и необходимый размер выборки.

Нужен, чтобы MDE в PLAN.md был не «прикинул», а воспроизводимым расчётом:
та же формула, что в отчёте, и CLI, которым можно проверить план до запуска.

    python -m experiment.power --p1 0.448 --mde 0.009
    python -m experiment.power --p1 0.448 --n 45000
"""

from __future__ import annotations

import argparse
import math

from scipy import stats


def z_alpha(alpha: float, two_sided: bool = True) -> float:
    return stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)


def z_beta(power: float) -> float:
    return stats.norm.ppf(power)


def mde_proportions(p1: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный детектируемый эффект (абсолютный, в долях) для доли p1.

    Стандартная формула для двух долей при равных группах:
    MDE = (z_{1−α/2} + z_{power}) · sqrt(2·p̄·(1−p̄)/n), где p̄ ≈ p1.
    """
    if not 0 < p1 < 1:
        raise ValueError("p1 должен быть между 0 и 1")
    if n_per_arm <= 0:
        raise ValueError("размер группы должен быть положительным")
    p_bar = p1
    return (z_alpha(alpha) + z_beta(power)) * math.sqrt(2 * p_bar * (1 - p_bar) / n_per_arm)


def required_sample(p1: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Сколько нужно на группу, чтобы поймать эффект mde (абсолютный)."""
    if mde <= 0:
        raise ValueError("MDE должен быть положительным")
    p_bar = p1
    numerator = 2 * (z_alpha(alpha) + z_beta(power)) ** 2 * p_bar * (1 - p_bar)
    return int(math.ceil(numerator / mde**2))


def relative_to_absolute(p1: float, relative: float) -> float:
    """Относительный эффект (доля от базовой) в абсолютные пункты."""
    return p1 * relative


def main() -> None:
    parser = argparse.ArgumentParser(description="MDE и размер выборки для A/B")
    parser.add_argument("--p1", type=float, required=True, help="базовая доля, например 0.448")
    parser.add_argument("--mde", type=float, help="эффект в абсолютных долях, например 0.009")
    parser.add_argument("--n", type=int, help="размер группы, если считаем MDE")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args()

    if args.n:
        mde = mde_proportions(args.p1, args.n, args.alpha, args.power)
        print(f"при n = {args.n:,} на группу MDE = {mde * 100:.2f} п.п. "
              f"(относительно {mde / args.p1 * 100:.1f} % от базы)")
    elif args.mde:
        n = required_sample(args.p1, args.mde, args.alpha, args.power)
        print(f"для эффекта {args.mde * 100:.2f} п.п. нужно {n:,} на группу "
              f"({2 * n:,} всего)")
    else:
        parser.error("нужен либо --n (считать MDE), либо --mde (считать выборку)")


if __name__ == "__main__":
    main()
