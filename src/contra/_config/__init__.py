# SPDX-FileCopyrightText: 2024-present Christopher Rowe <chris.rowe19@outlook.com>
#
# SPDX-License-Identifier: None
import os
from typing import Any, Dict

from QuasarCode.IO.Configurations import YamlConfig

_filepath = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

class ContraSettings(YamlConfig):
    """
    Singleton settings object. Call constructor to get instance.

    Attributes:
        L_star_mass_of_z: Dict[float, float]
    """

    instance = None
#    def __new__(mcls: type["ContraSettings"], name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> "ContraSettings":
    def __new__(cls, *args, **kwargs: Any) -> "ContraSettings":
        if ContraSettings.instance == -1:
            # Constructor has been explicitley enabled
#            return super().__new__(name, bases, namespace, **kwargs)
            return super().__new__(*args, **kwargs)
        else:
            if ContraSettings.instance is None:
                ContraSettings.instance = -1 # Allow the constructor to create a new instance
                if os.path.exists(_filepath):
                    ContraSettings.instance = ContraSettings(
                        filepath = _filepath,
                        safe_load = False
                    )
                else:
                    ContraSettings.instance = ContraSettings.create_default()
            return ContraSettings.instance
        
    def __init__(self, *args, **kwargs):
        super().__init__(True, *args, **kwargs)

    def save(self):
        with open(_filepath, "w") as file:
            file.write(str(self))

    @staticmethod
    def create_default() -> "ContraSettings":
        properties = ContraSettings()
        properties.L_star_mass_of_z = { # in Msun
            0.0  : -1,
            0.1  : -1,
            0.5  : -1,
            1.0  : -1,
            2.0  : -1,
            3.0  : -1,
            4.0  : -1,
            5.0  : -1,
            6.0  : -1,
            7.0  : -1,
            8.0  : -1,
            9.0  : -1,
            10.0 : -1
        }#TODO: set default values
        properties.save()
        return properties
    
    @staticmethod
    def refresh() -> bool:
        tmp = ContraSettings.instance
        ContraSettings.instance = None
        try:
            ContraSettings()
            return True
        except:
            ContraSettings.instance = tmp
            return False
