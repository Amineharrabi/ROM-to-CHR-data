import struct 
def parse_nes_header(path) :
    with open(path,'rb') as f :
        header = f.read(16) 
        if header[:4] != b'NES\x1a':
            raise ValueError("Not a valid NES file")
        prg_size = header[4] * 16 * 1024
        chr_size = header[5] * 8 * 1024
        mapper = (header[6] >> 4) | (header[7] & 0xF0)

        return {
            "prg_size": prg_size,
            "chr_size": chr_size,
            "mapper": mapper
        }

print(parse_nes_header("roms/MicroMages.nes"))