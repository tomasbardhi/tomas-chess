def print_channel(channel, name="", end='\n'):
    # name of channel
    print(name)
    # channel
    for rank in range(8):
        for file in range(8):
            print('1' if channel[rank, file] else '.', end=' ')
        print()
    print(end, end='')

def print_channels(input):

    # function to print channels next to each other
    def print_group(channels, names):
        for rank in range(8):
            for channel_index in channels:
                for file in range(8):
                    print('1' if input[channel_index, rank, file] else '.', end=' ')
                print('\t', end='')
            print()
        for name in names:
            print(f"{name:<8}", end='\t\t')
        print('\n')

    # names for each channel
    piece_names = ['Pawn', 'Knight', 'Bishop', 'Rook', 'Queen', 'King']
    white_pieces_names = ['White ' + name for name in piece_names]
    black_pieces_names = ['Black ' + name for name in piece_names]
    special_names = [
        "Turn", 
        "W QSide", "W KSide", 
        "B QSide", "B KSide", 
        "50-Move", 
        "En Passant"
    ]

    # print white and black piece channels
    print()
    print("White pieces:")
    print_group(range(6), white_pieces_names)
    print("Black pieces:")
    print_group(range(6, 12), black_pieces_names)

    # print special channels
    print("Special channels:")
    print_group(range(12, 19), special_names)