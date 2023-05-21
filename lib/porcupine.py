import platform

def wakeup_keywords():
    keywords = []
    if platform.system() == "Darwin":
        keywords = ["./wakeup_word_models/chat-g-p-t_en_mac_v2_2_0.ppn"]
    elif platform.machine() == "armv7l":
        keywords = [
            "./wakeup_word_models/chat-g-p-t_en_raspberry-pi_v2_2_0.ppn",
#            "./wakeup_word_models/chat-g-p-t_pt_raspberry-pi_v2_2_0.ppn",
        ]
    else:
        raise "OS not supported, only macOS and Raspberry PI are supported right now"

    return keywords
