import pint;  # units
unit = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    system=None,
    auto_reduce_dimensions=True, )


m_mol_materials = {
    'AIR': 28.9645E-03 * unit('kg / mol'),  # dry atmospheric air
    'N2':  28.006E-03 * unit('kg / mol'),
    'CO2': 44.01E-03 * unit('kg / mol'),
    'CO':  28.01E-03 * unit('kg / mol'),
}


def get_Rfluid(m_mol=28.9645E-03 * unit('kg / mol')):
    Runiv = 8.3145 * unit('J / (mol * K)')
    return Runiv / m_mol


def get_R_cp(material, k):
    """
    :param material: e.g. material='air'
    :param k: e.g. k=1.4
    :return:
    """
    if material in m_mol_materials:
        Rfluid = get_Rfluid(m_mol=m_mol_materials[material])
    else:
        NotImplementedError('Material {} not implemented. Calculate R and cp manually.'.format(material))
    cp = Rfluid * k / (k - 1)
    return Rfluid, cp