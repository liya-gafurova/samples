import json
from typing import Optional, List, Any, Tuple

import torch
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

from text_generation.source_parser import Example

FILEPATH = '/home/lg/PycharmProjects/my/samples/text_generation/out_all.json'


class DataExample(BaseModel):
    source: Optional[str]
    target: Optional[str]


def upload_data(filepath: str) -> list[Example]:
    """
    upload dataset with list[Example] structure (obtained with source_parser.parse_files())
    :param filepath: dataset filepath
    :return: list[Examples]
    """
    datas = []

    with open(filepath, 'r') as file:
        raw = file.read()
        data = json.loads(raw)

        exmps = []
        for obj in data:
            exmp: Example = Example.parse_obj(obj)
            d = DataExample(
                source=f'<title>: {exmp.title}, <description>: {exmp.description}, <keywords>: {", ".join(exmp.keywords)}',
                target=f'{exmp.text.text}'
            )
            datas.append(d.dict())

    with open('./dataset.json', 'w') as out:
        out.write(json.dumps(datas))

    return exmps


def count_data_len(filepath: str, field: str) -> Tuple[str, int]:
    """
    :param filepath: filepath with dataset
    :param field: particular string field of dataset to count the char length
    :return: field name, average char len of value in 'field'
    """
    all_lengths: int = 0
    objs_count: int = 0

    with open(filepath, 'r') as file:
        raw = file.read()
        data: dict = json.loads(raw)

        for d in data:
            field_len = len(d[field])
            all_lengths += field_len
            objs_count += 1

    avg_char_count = int(all_lengths / objs_count)
    print(f'Avg length of field {field}: {avg_char_count}')

    return field, avg_char_count


def enrich_dataset(filepath: str, new_dataset_filepath: str) -> int:
    """
    old dataset:  [
        {
        "source": "<title>: Parimatch Casino India | Get 150% up to \u20b9105,000 for signing up, <description>: Parimatch Online Casino offers: 7,000 licensed slots, over 15 deposit and withdrawal methods, a wide selection of Indian games: Teen Patti, Andar Bahar, Namaste Roulette, Cricket War., <keywords>: parimatch,  parimatch casino",
        "target": "<Parimatch Casino Overview\n  \n\nParimatch online casino is a Curacao eGaming licensed leader in the gambling industry. The casino has over 1,000,000 users and a gaming library of more than 7,000 slots. The collection features developments from top providers: 1x2 Gaming, EGT, Microgaming, NetEnt, Endorphina, Novomatic, Tom Horn. Join Parimatch now and get a welcome bonus of 150% up to \u20b9 105,000.\nHow to Start Playing in Parimatch?\nThere's no need to register to check out the range of games and try out the slots you like. The Slots section features thousands of slots that you can play in demo mode. Playing for virtual coins, the gambler cannot cash out his winnings. To do so, it is necessary to join an online casino.\n\n\nTo become a full-fledged Parimatch user, you need to take three steps:\n1. Register. Go to the website and click \"Sign up\" at the top of the page. Type in your phone number, come up with a password and click on the sign up button.\n2. Deposit account. Click \"Deposit\" on the top menu, choose a payment method and create a deposit of \u20b9350 to qualify for the welcome bonus.\n3. Proceed to play. Select the casino section and start any of the slot machines you like by clicking on the 'Play' button.\n\n\nTo find entertainment, there are handy filters sorted by category, provider and the popularity of the machines. Add video slots to your favourite list by clicking on the dedicated icon at the top of the game title.\nWelcome Bonus for Beginners\nAt the time Parimatch casino review was compiled, new users were offered a generous bonus of 150% up to \u20b9105,000. To join the promo, you must:\n* Create an account on the casino website;\n* Confirm your phone number;\n* Make your first deposit of \u20b9350.\n\n\nThe credit money is credited to the bonus account and requires wagering with a wager of \u00d730. For example, if the bonus was \u20b93,000, the user can withdraw it when they spend \u20b990,000 on wagers. The wagering requirements must be met within 30 days, otherwise the bonus will be cancelled.\nParimatch Game Assortment\nThe Parimatch online casino offers over 7,000 slots, which can be found in four sections: Live Casino, Slots, Instant Games and TV Games. The collection includes original games from 80+ official providers, including many industry leaders: Booongo, Netgame, Play'n'Go, Amazing Gaming, EGT, Betsoft, Evoplay and OneTouch. All popular entertainment formats are available, including live dealers, instant raffles, scratch cards, slot machines and TV lotteries.\nRoulette\nParimatch has over 20 different Roulette products; betting on rows, sectors, red/black, odd and even numbers:\n* Roulette X5;\n* Sapphire Roulette;\n* Multifire Roulette;\n* American Roulette 3D;\n* Gamevy Roulette.\n\n\nThe modern versions of the table game differ from the classics in their betting options. In some of them, the player can return 50% of the amount wagered, even if the ball stops on a zero cell.\nSlots\nQuality graphics, a wide range of options and high betting returns are key advantages of Parimatch's slots games. There are just over 6,000 titles divided into dozens of categories:\n* Fruit;\n* Books;\n* 777;\n* Asia;\n* Crystals;\n* Egypt;\n* Football, etc.\n\n\nThere are classic 3-reel machines, models with progressive jackpots and betting multipliers. A lot of slots with 3D graphics, Megaways mechanics and bonuses: freespins, risk game.\nBlackjack\nThe card game attracts not only with its simple rules, but also with its high chances of winning. In blackjack, the casino's mathematical advantage is minimal, as almost all experienced users know. Parimatch has many variations of the game, but the most popular ones are:\n* Blackjack MH Perfect Pairs;\n* Blackjack Side Bets;\n* Hi-Lo Blackjack (5 Box);\n* Blackjack MH 21+3;\n* Perfect Strategy Blackjack;\n* Blackjack 3 Hand.\nSome versions of blackjack include additional features such as hedge betting and splitting the cards into two or three hands.\nLive Casino\nLive dealer section features over 300 tables with games of Andar Bahar, Baccarat, Poker, Roulette, Sic Bo, Monopoly and Lottery. The collection features the most popular developments from Ezugi, NetEnt, Authentic Gaming, Evolution Gaming and other vendors. All of them are sorted into tabs so that the user can find the game of interest in a couple of clicks.\n\n\nThe welcome page contains the most sought-after entertainment:\n* Craps Live;\n* Single Poker;\n* Monopoly Live;\n* Namaste Roulette;\n* Andar Bahar;\n* Crazy Time;\n* Lightning Roulette.\n\n\nPlayers can enjoy a casino-like atmosphere and a highly interactive experience. If you want to test your luck and get a positive feeling, try the games in real time.\nMobile App Parimatch Casino\nTo play slots from your phone, we recommend installing the Parimatch mobile app. Android device owners can download the software directly from the casino website. If you have an iPhone, download the software from the AppStore.\n\n\nParimatch app supports the same games featured on the website. Authorised users can play casino games, bet on sports, deposit and withdraw money, participate in promotions and tournaments, and contact customer support.\n\n\nInstallation of the mobile app is done in three steps:\n1. Open the website in your mobile browser;\n2. Click on \"Apps Android/iOS\" in the side menu;\n3. Download and launch the software by clicking on the \"Install\" button.\n4. Many options are available even in guest mode. But to start playing for money, you have to register or log in.\nAdvantages of Online Casino Parimatch\nParimatch is a well known brand in the gambling world and is trusted by hundreds of thousands of players from all over Asia and Europe. Registered players can count on fair gaming conditions, impeccable service, and quick withdrawal requests. Among all the advantages of the online casino are worth mentioning:\n* A huge collection of games with excellent gameplay;\n* A friendly support team;\n* A wide selection of Indian payment systems;\n* Free app for Android and iOS;\n* Generous bonuses and promotions.\n\n\nThe company operates under an international license, which confirms its legality in India. Join Parimatch and get rewarded for registration, deposits, active play. >"
     },
      where "target" text is of average 8000 characters is too long for T5 generation fine-tuning

     this method converts old dataset structure to new (71 data points, where 1 target is 1 sentence from original target). Note, that new token <prompt> added:
     [
        {
        "source": "<title>: Parimatch Casino India | Get 150% up to \u20b9105,000 for signing up, <description>: Parimatch Online Casino offers: 7,000 licensed slots, over 15 deposit and withdrawal methods, a wide selection of Indian games: Teen Patti, Andar Bahar, Namaste Roulette, Cricket War., <keywords>: parimatch,  parimatch casino",
        "target": "<Parimatch Casino Overview"
    },
    {
        "source": "<title>: Parimatch Casino India | Get 150% up to \u20b9105,000 for signing up, <description>: Parimatch Online Casino offers: 7,000 licensed slots, over 15 deposit and withdrawal methods, a wide selection of Indian games: Teen Patti, Andar Bahar, Namaste Roulette, Cricket War., <keywords>: parimatch,  parimatch casino, <prompt>: <Parimatch Casino Overview",
        "target": "Parimatch online casino is a Curacao eGaming licensed leader in the gambling industry. The casino has over 1,000,000 users and a gaming library of more than 7,000 slots. The collection features developments from top providers: 1x2 Gaming, EGT, Microgaming, NetEnt, Endorphina, Novomatic, Tom Horn. Join Parimatch now and get a welcome bonus of 150% up to \u20b9 105,000."
    },
    ]
    :param filepath: filepath of a dataset that will be enriched
    :param new_dataset_filepath: filepath of new enriched dataset
    :return:
    """
    with open(filepath) as file:
        raw = file.read()
        data = json.loads(raw)

    objs: list[DataExample] = [DataExample(**d) for d in data]
    new_objs = []
    for obj in objs:
        sentences = obj.target.split('\n')
        sentences = list(filter(lambda s: s not in ['', ' ', '  ', '\t'], sentences))

        for i, sentence in enumerate(sentences):
            new_data = DataExample()
            if i == 0:
                new_data.source = obj.source
                new_data.target = sentence
            else:
                new_data.source = obj.source + f', <prompt>: {sentences[i - 1]}'
                new_data.target = sentence

            new_objs.append(new_data.dict())

    with open(new_dataset_filepath, 'w') as out:
        out.write(json.dumps(new_objs))

    return len(new_objs)


def generate_text(filepath: Optional[str] = None):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    prompt_base = """summarize: Parimatch Casino Overview\n  \n\nParimatch online casino is a Curacao eGaming licensed leader in the gambling industry. The casino has over 1,000,000 users and a gaming library of more than 7,000 slots. The collection features developments from top providers: 1x2 Gaming, EGT, Microgaming, NetEnt, Endorphina, Novomatic, Tom Horn. Join Parimatch now and get a welcome bonus of 150% up to \u20b9 105,000.\nHow to Start Playing in Parimatch?\nThere's no need to register to check out the range of games and try out the slots you like. The Slots section features thousands of slots that you can play in demo mode. Playing for virtual coins, the gambler cannot cash out his winnings. To do so, it is necessary to join an online casino.\n\n\nTo become a full-fledged Parimatch user, you need to take three steps:\n1. Register. Go to the website and click \"Sign up\" at the top of the page. Type in your phone number, come up with a password and click on the sign up button.\n2. Deposit account. Click \"Deposit\" on the top menu, choose a payment method and create a deposit of \u20b9350 to qualify for the welcome bonus.\n3. Proceed to play. Select the casino section and start any of the slot machines you like by clicking on the 'Play' button.\n\n\nTo find entertainment, there are handy filters sorted by category, provider and the popularity of the machines. Add video slots to your favourite list by clicking on the dedicated icon at the top of the game title.\nWelcome Bonus for Beginners\nAt the time Parimatch casino review was compiled, new users were offered a generous bonus of 150% up to \u20b9105,000. To join the promo, you must:\n* Create an account on the casino website;\n* Confirm your phone number;\n* Make your first deposit of \u20b9350.\n\n\nThe credit money is credited to the bonus account and requires wagering with a wager of \u00d730. For example, if the bonus was \u20b93,000, the user can withdraw it when they spend \u20b990,000 on wagers. The wagering requirements must be met within 30 days, otherwise the bonus will be cancelled.\nParimatch Game Assortment\nThe Parimatch online casino offers over 7,000 slots, which can be found in four sections: Live Casino, Slots, Instant Games and TV Games. The collection includes original games from 80+ official providers, including many industry leaders: Booongo, Netgame, Play'n'Go, Amazing Gaming, EGT, Betsoft, Evoplay and OneTouch. All popular entertainment formats are available, including live dealers, instant raffles, scratch cards, slot machines and TV lotteries.\nRoulette\nParimatch has over 20 different Roulette products; betting on rows, sectors, red/black, odd and even numbers:\n* Roulette X5;\n* Sapphire Roulette;\n* Multifire Roulette;\n* American Roulette 3D;\n* Gamevy Roulette.\n\n\nThe modern versions of the table game differ from the classics in their betting options. In some of them, the player can return 50% of the amount wagered, even if the ball stops on a zero cell.\nSlots\nQuality graphics, a wide range of options and high betting returns are key advantages of Parimatch's slots games. There are just over 6,000 titles divided into dozens of categories:\n* Fruit;\n* Books;\n* 777;\n* Asia;\n* Crystals;\n* Egypt;\n* Football, etc.\n\n\nThere are classic 3-reel machines, models with progressive jackpots and betting multipliers. A lot of slots with 3D graphics, Megaways mechanics and bonuses: freespins, risk game.\nBlackjack\nThe card game attracts not only with its simple rules, but also with its high chances of winning. In blackjack, the casino's mathematical advantage is minimal, as almost all experienced users know. Parimatch has many variations of the game, but the most popular ones are:\n* Blackjack MH Perfect Pairs;\n* Blackjack Side Bets;\n* Hi-Lo Blackjack (5 Box);\n* Blackjack MH 21+3;\n* Perfect Strategy Blackjack;\n* Blackjack 3 Hand.\nSome versions of blackjack include additional features such as hedge betting and splitting the cards into two or three hands.\nLive Casino\nLive dealer section features over 300 tables with games of Andar Bahar, Baccarat, Poker, Roulette, Sic Bo, Monopoly and Lottery. The collection features the most popular developments from Ezugi, NetEnt, Authentic Gaming, Evolution Gaming and other vendors. All of them are sorted into tabs so that the user can find the game of interest in a couple of clicks.\n\n\nThe welcome page contains the most sought-after entertainment:\n* Craps Live;\n* Single Poker;\n* Monopoly Live;\n* Namaste Roulette;\n* Andar Bahar;\n* Crazy Time;\n* Lightning Roulette.\n\n\nPlayers can enjoy a casino-like atmosphere and a highly interactive experience. If you want to test your luck and get a positive feeling, try the games in real time.\nMobile App Parimatch Casino\nTo play slots from your phone, we recommend installing the Parimatch mobile app. Android device owners can download the software directly from the casino website. If you have an iPhone, download the software from the AppStore.\n\n\nParimatch app supports the same games featured on the website. Authorised users can play casino games, bet on sports, deposit and withdraw money, participate in promotions and tournaments, and contact customer support.\n\n\nInstallation of the mobile app is done in three steps:\n1. Open the website in your mobile browser;\n2. Click on \"Apps Android/iOS\" in the side menu;\n3. Download and launch the software by clicking on the \"Install\" button.\n4. Many options are available even in guest mode. But to start playing for money, you have to register or log in.\nAdvantages of Online Casino Parimatch\nParimatch is a well known brand in the gambling world and is trusted by hundreds of thousands of players from all over Asia and Europe. Registered players can count on fair gaming conditions, impeccable service, and quick withdrawal requests. Among all the advantages of the online casino are worth mentioning:\n* A huge collection of games with excellent gameplay;\n* A friendly support team;\n* A wide selection of Indian payment systems;\n* Free app for Android and iOS;\n* Generous bonuses and promotions.\n\n\nThe company operates under an international license, which confirms its legality in India. Join Parimatch and get rewarded for registration, deposits, active play. """
    prompt = prompt_base

    results = []

    with torch.no_grad():
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, do_sample=True,
                             max_length=128,
                             top_k=50,
                             top_p=0.95,)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f'Source: {len(prompt_base)}, summary: {len(result)}')

        print(result)


    print('. '.join(results))


generate_text('ergeth')
