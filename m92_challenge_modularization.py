def celcius_2_kelvin(celsius: float) -> tuple[float, float]:
    # Convert Celsius to Kelvin
    kelvin = celsius + 273.15
    if kelvin < 0:
        raise ValueError(f"Invalid temperature, below absolute zero: kelvin {kelvin} < 0")
    else:
        fahrenheit = (celsius * (9 / 5)) + 32
        if fahrenheit < -459.67:
            raise ValueError(f"Invalid temperature, below absolute zero: fahrenheit {fahrenheit} < -459.67")
    return celsius, kelvin

# write a test for the function
# TODO...

def convert_temperature(temperature, unit):
    if unit == "F":
        # Convert Fahrenheit to Celsius
        celsius = (temperature - 32) * (5 / 9)
        if celsius < -273.15:
            # Invalid temperature, below absolute zero
            return "Invalid temperature"
        else:
            celcius_2_kelvin(temperature)
    elif unit == "C":
        # Convert Celsius to Fahrenheit
        fahrenheit = (temperature * (9 / 5)) + 32
        if fahrenheit < -459.67:
            # Invalid temperature, below absolute zero
            return "Invalid temperature"
        else:
            # Convert Celsius to Kelvin
            kelvin = temperature + 273.15
            if kelvin < 0:
                # Invalid temperature, below absolute zero
                return "Invalid temperature"
            else:
                return fahrenheit, kelvin
    elif unit == "K":
        # Convert Kelvin to Celsius
        celsius = temperature - 273.15
        if celsius < -273.15:
            # Invalid temperature, below absolute zero
            return "Invalid temperature"
        else:
            # Convert Celsius to Fahrenheit
            fahrenheit = (celsius * (9 / 5)) + 32
            if fahrenheit < -459.67:
                # Invalid temperature, below absolute zero
                return "Invalid temperature"
            else:
                return celsius, fahrenheit
    else:
        return "Invalid unit"
