from enum import Enum


class Item(Enum):
    BOTTOMBUN = "bottombun"
    LETTUCE = "lettuce"
    PATTY = "patty"
    TOPBUN = "topbun"
    BREAD = "bread"
    CHEESE = "cheese"
    TOMATO = "tomato"
    ONION = "onion"
    CHICKEN = "chicken"
    MEDICINE = "medicine"
    PULSE_CHECKER = "pulse_checker-equipment"
    AED = "aed"
    CPR_KIT = "cpr_kit"
    CPR_BOARD = "cpr_board"
    # VENTILATOR = "ventilator"
    CPR_STOOL = "cpr_stool"
    PATIENT = "patient"
    SYRINGE = "syringe"
    CPR_PUMP = "pump"    

class Player(Enum):
    ROBOT = "robot"
    NURSE = "nurse"


class Station(Enum):
    BOARD = "board"
    STOVE = "stove"
    TABLE = "table"
    PATIENT_LEGS = "patient_legs"
    PATIENT_BED_STATION = "patient_bed_station"
    HOSP_CART_LEFT = "hospital_cart_left"
    HOSP_CART_RIGHT = "hospital_cart_right"
    HOSP_CART = "hospital_cart"


def str_to_typed_enum(s):
    """`
    Attempts to convert a string into any of the typed enums.
    Args:
        s (str): String to convert.

    Raises:
        ValueError: If the string cannot be converted into any of the typed enums.

    Returns:
        typed_enum (Enum): Enum of the string.
    """
    for typed_enum in [Item, Player, Station]:
        try:
            return typed_enum(s)
        except ValueError:
            pass
    raise ValueError(f"Could not convert {s} into any of the typed enums.")
