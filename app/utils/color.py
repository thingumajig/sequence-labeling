from hashlib import md5
import colorsys


def getHexComponent(hex, i, normalized=False):
    comp = int(hex[i : i + 2], 16)
    if normalized:
        comp = comp / 255.0
    return comp


def getColorForText(tag):
    hex = md5(tag.encode("utf8")).hexdigest()[0:6]
    # return f"#{hex}"
    rgb = hex2rgb(hex, True)
    hsv = list(colorsys.rgb_to_hsv(*rgb))
    hsv[1] *= 0.3
    rgb = colorsys.hsv_to_rgb(*hsv)
    return rgb2hex(rgb, True)


def hex2rgb(hex, normalized=False):
    return [getHexComponent(hex, i, normalized) for i in (0, 2, 4)]


def rgb2hex(rgb, fromNormalized=False):
    if fromNormalized:
        rgb = [int(v * 255.0) for v in rgb]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" % rgb
