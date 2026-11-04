# Language review sheet

Every user request and agent reply template, for native-speaker review.
Placeholders like `{amount}` are filled in automatically. Reply with the template id and your fix.
Rows marked **test** are held out from training and used only for evaluation.

## User requests

### airtime_direct

| id | lang | split | text |
|---|---|---|---|
| `airtime_direct.pcm.0` | Pidgin | train | Abeg buy {amount} {network} airtime for {phone}. |
| `airtime_direct.pcm.1` | Pidgin | train | Make you recharge {phone} with {amount}, na {network} line. |
| `airtime_direct.pcm.2` | Pidgin | train | Abeg load {amount} card for my {network} number {phone}. |
| `airtime_direct.pcm.3` | Pidgin | train | I wan buy airtime of {amount} for {phone}, {network}. |
| `airtime_direct.pcm.4` | Pidgin | train | Help me top up {phone} ({network}) with {amount} abeg. |
| `airtime_direct.pcm.5` | Pidgin | test | Oya send {amount} {network} recharge enter {phone}. |
| `airtime_direct.yo.0` | Yoruba | train | Ẹ jọ̀ọ́, ẹ bá mi ra káàdì {network} {amount} fún {phone}. |
| `airtime_direct.yo.1` | Yoruba | train | Bá mi ra airtime {amount} fún nọ́ńbà {phone}, {network} ni. |
| `airtime_direct.yo.2` | Yoruba | train | Mo fẹ́ ra káàdì ìpè {amount} fún {phone} ({network}). |
| `airtime_direct.yo.3` | Yoruba | train | Jọ̀ọ́ fi airtime {amount} ránṣẹ́ sí {phone}, {network} ni. |
| `airtime_direct.yo.4` | Yoruba | test | Ra airtime {network} {amount} fún mi sórí {phone}. |
| `airtime_direct.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ bá mi buy {amount} airtime for {phone}, {network} ni. |
| `airtime_direct.yo_en.1` | Yoruba-English mix | train | Ẹ bá mi recharge {phone} with {amount}, {network} line ni. |
| `airtime_direct.yo_en.2` | Yoruba-English mix | train | Mo fẹ́ top up {phone} pẹ̀lú {amount}, {network} ni network ẹ̀. |
| `airtime_direct.yo_en.3` | Yoruba-English mix | test | Buy {network} card {amount} fún {phone} jọ̀ọ́. |
| `airtime_direct.en.0` | English | train | Buy {amount} {network} airtime for {phone}. |
| `airtime_direct.en.1` | English | train | Please recharge {phone} with {amount} airtime. It's {network}. |
| `airtime_direct.en.2` | English | train | Can you top up my {network} line {phone} with {amount}? |
| `airtime_direct.en.3` | English | train | I need {amount} airtime on {phone} ({network}). |
| `airtime_direct.en.4` | English | train | Send {amount} {network} recharge to {phone} please. |
| `airtime_direct.en.5` | English | test | Top up {phone} {amount}, network is {network} |

### data_direct

| id | lang | split | text |
|---|---|---|---|
| `data_direct.pcm.0` | Pidgin | train | Abeg buy {plan} data for {phone}, na {network}. |
| `data_direct.pcm.1` | Pidgin | train | Make you load {plan} {network} data for my line {phone}. |
| `data_direct.pcm.2` | Pidgin | train | I wan subscribe {plan} data for {phone} ({network}). |
| `data_direct.pcm.3` | Pidgin | train | My data don finish o, abeg buy {plan} for {phone}, {network} line. |
| `data_direct.pcm.4` | Pidgin | test | Oya buy {network} {plan} for {phone} sharp sharp. |
| `data_direct.yo.0` | Yoruba | train | Ẹ jọ̀ọ́ ẹ bá mi ra data {plan} fún {phone}, {network} ni. |
| `data_direct.yo.1` | Yoruba | train | Data mi ti tán, bá mi ra {plan} sórí {phone} ({network}). |
| `data_direct.yo.2` | Yoruba | train | Mo fẹ́ ra data {network} {plan} fún nọ́ńbà {phone}. |
| `data_direct.yo.3` | Yoruba | test | Jọ̀ọ́ fi data {plan} sí {phone}, {network} ni. |
| `data_direct.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ buy {plan} data for {phone}, {network} ni. |
| `data_direct.yo_en.1` | Yoruba-English mix | train | Data mi ti finish, abeg load {plan} on {phone} ({network}). |
| `data_direct.yo_en.2` | Yoruba-English mix | test | Ẹ bá mi subscribe {plan} data fún {phone}, {network} line ni. |
| `data_direct.en.0` | English | train | Buy {plan} {network} data for {phone}. |
| `data_direct.en.1` | English | train | Please get me a {plan} data bundle on {phone} ({network}). |
| `data_direct.en.2` | English | train | Subscribe {phone} to {plan} data, the line is {network}. |
| `data_direct.en.3` | English | train | I need {plan} of data on my {network} number {phone}. |
| `data_direct.en.4` | English | test | Load {plan} data on {phone}, network {network}. |

### electricity

| id | lang | split | text |
|---|---|---|---|
| `electricity.pcm.0` | Pidgin | train | Abeg pay {amount} for my light, meter na {meter}, {disco}. |
| `electricity.pcm.1` | Pidgin | train | My light units don finish, abeg buy {amount} for meter {meter}, {disco}. |
| `electricity.pcm.2` | Pidgin | train | Help me recharge my meter {meter} with {amount}, na {disco}. |
| `electricity.pcm.3` | Pidgin | test | I wan pay light bill {amount}, {disco} meter {meter}. |
| `electricity.yo.0` | Yoruba | train | Ẹ jọ̀ọ́ ẹ bá mi san owó iná {amount}, nọ́ńbà mítà mi ni {meter}, {disco}. |
| `electricity.yo.1` | Yoruba | train | Iná mi ti tán, bá mi ra iná {amount} fún mítà {meter} ({disco}). |
| `electricity.yo.2` | Yoruba | test | Mo fẹ́ san {amount} fún iná mànàmáná. {disco} ni, mítà {meter}. |
| `electricity.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ pay {amount} for light, meter mi ni {meter}, {disco}. |
| `electricity.yo_en.1` | Yoruba-English mix | train | Units mi ti tán o, buy {amount} for meter {meter} ({disco}). |
| `electricity.yo_en.2` | Yoruba-English mix | test | Ẹ bá mi recharge meter {meter} pẹ̀lú {amount}, {disco} ni. |
| `electricity.en.0` | English | train | Pay {amount} for my electricity. Meter number {meter}, {disco}. |
| `electricity.en.1` | English | train | Buy {amount} light units for meter {meter} on {disco}. |
| `electricity.en.2` | English | train | Please recharge my prepaid meter {meter} ({disco}) with {amount}. |
| `electricity.en.3` | English | test | I want to buy electricity worth {amount}. {disco}, meter {meter}. |

### airtime_contact

| id | lang | split | text |
|---|---|---|---|
| `airtime_contact.pcm.0` | Pidgin | train | Abeg send {amount} airtime give {contact}. |
| `airtime_contact.pcm.1` | Pidgin | train | Buy {amount} card for {contact} abeg. |
| `airtime_contact.pcm.2` | Pidgin | train | Make you recharge {contact} line with {amount}. |
| `airtime_contact.pcm.3` | Pidgin | train | Abeg dash {contact} {amount} card. |
| `airtime_contact.pcm.4` | Pidgin | test | {contact} say im no get credit, abeg send am {amount}. |
| `airtime_contact.yo.0` | Yoruba | train | Ẹ bá mi fi airtime {amount} ránṣẹ́ sí {contact}. |
| `airtime_contact.yo.1` | Yoruba | train | Ra káàdì {amount} fún {contact}. |
| `airtime_contact.yo.2` | Yoruba | train | Fún {contact} ní káàdì {amount}. |
| `airtime_contact.yo.3` | Yoruba | test | {contact} ò ní káàdì, fi {amount} ránṣẹ́ sí i. |
| `airtime_contact.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ send {amount} airtime sí {contact}. |
| `airtime_contact.yo_en.1` | Yoruba-English mix | train | Buy {amount} card fún {contact} abeg. |
| `airtime_contact.yo_en.2` | Yoruba-English mix | train | Give {contact} {amount} airtime jọ̀ọ́. |
| `airtime_contact.yo_en.3` | Yoruba-English mix | test | Ẹ bá mi recharge line {contact} with {amount}. |
| `airtime_contact.en.0` | English | train | Send {amount} airtime to {contact}. |
| `airtime_contact.en.1` | English | train | Buy {amount} recharge card for {contact}. |
| `airtime_contact.en.2` | English | train | Top up {contact}'s line with {amount}. |
| `airtime_contact.en.3` | English | train | Give {contact} {amount} airtime. |
| `airtime_contact.en.4` | English | test | Please get {amount} airtime for {contact}. |

### data_contact

| id | lang | split | text |
|---|---|---|---|
| `data_contact.pcm.0` | Pidgin | train | Abeg buy {plan} data give {contact}. |
| `data_contact.pcm.1` | Pidgin | train | {contact} data don finish, load {plan} for am. |
| `data_contact.pcm.2` | Pidgin | test | Make you send {plan} data to {contact}. |
| `data_contact.yo.0` | Yoruba | train | Bá mi ra data {plan} fún {contact}. |
| `data_contact.yo.1` | Yoruba | train | Data {contact} ti tán, ra {plan} fún un. |
| `data_contact.yo.2` | Yoruba | test | Jọ̀ọ́ fi data {plan} ránṣẹ́ sí {contact}. |
| `data_contact.yo_en.0` | Yoruba-English mix | train | Buy {plan} data fún {contact} jọ̀ọ́. |
| `data_contact.yo_en.1` | Yoruba-English mix | train | Data {contact} ti finish, load {plan} for am. |
| `data_contact.yo_en.2` | Yoruba-English mix | test | Ẹ bá mi send {plan} data sí {contact}. |
| `data_contact.en.0` | English | train | Buy {plan} data for {contact}. |
| `data_contact.en.1` | English | train | Get {contact} a {plan} data bundle. |
| `data_contact.en.2` | English | test | Please load {plan} on {contact}'s phone. |

### transfer_direct

| id | lang | split | text |
|---|---|---|---|
| `transfer_direct.pcm.0` | Pidgin | train | Abeg send {amount} to {acct}, {bank}. |
| `transfer_direct.pcm.1` | Pidgin | train | Transfer {amount} enter {bank} account {acct}. |
| `transfer_direct.pcm.2` | Pidgin | train | Make you pay {amount} for this account: {acct} {bank}. |
| `transfer_direct.pcm.3` | Pidgin | test | I wan send {amount} give person, na {bank} {acct}. |
| `transfer_direct.yo.0` | Yoruba | train | Ẹ jọ̀ọ́ fi {amount} ránṣẹ́ sí àkáǹtì {acct}, {bank}. |
| `transfer_direct.yo.1` | Yoruba | train | Bá mi san {amount} sí àkáǹtì {bank} {acct}. |
| `transfer_direct.yo.2` | Yoruba | test | Mo fẹ́ fi owó {amount} ránṣẹ́ sí {acct} ní {bank}. |
| `transfer_direct.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ transfer {amount} sí {acct}, {bank} ni. |
| `transfer_direct.yo_en.1` | Yoruba-English mix | train | Send {amount} sí account yìí: {acct} {bank}. |
| `transfer_direct.yo_en.2` | Yoruba-English mix | test | Ẹ bá mi pay {amount} into {bank} account {acct}. |
| `transfer_direct.en.0` | English | train | Send {amount} to {bank} account {acct}. |
| `transfer_direct.en.1` | English | train | Transfer {amount} to {acct} ({bank}). |
| `transfer_direct.en.2` | English | train | Please pay {amount} into {bank} {acct}. |
| `transfer_direct.en.3` | English | test | I want to transfer {amount}. Account number {acct}, {bank}. |

### transfer_contact

| id | lang | split | text |
|---|---|---|---|
| `transfer_contact.pcm.0` | Pidgin | train | Abeg send {amount} give {contact}. |
| `transfer_contact.pcm.1` | Pidgin | train | Transfer {amount} to {contact} account. |
| `transfer_contact.pcm.2` | Pidgin | train | Abeg dash {contact} {amount}. |
| `transfer_contact.pcm.3` | Pidgin | train | I owe {contact} {amount}, abeg settle am. |
| `transfer_contact.pcm.4` | Pidgin | train | Give {contact} {amount} make e use am for transport. |
| `transfer_contact.pcm.5` | Pidgin | test | {contact} need {amount}, abeg send am. |
| `transfer_contact.yo.0` | Yoruba | train | Fi {amount} ránṣẹ́ sí {contact}. |
| `transfer_contact.yo.1` | Yoruba | train | Bá mi san {amount} sí àkáǹtì {contact}. |
| `transfer_contact.yo.2` | Yoruba | train | Fún {contact} ní {amount}. |
| `transfer_contact.yo.3` | Yoruba | train | Mo jẹ {contact} ní {amount}, bá mi san án. |
| `transfer_contact.yo.4` | Yoruba | train | Bá mi fún {contact} ní owó {amount}. |
| `transfer_contact.yo.5` | Yoruba | test | {contact} nílò {amount}, jọ̀ọ́ fi ránṣẹ́ sí i. |
| `transfer_contact.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ send {amount} sí {contact}. |
| `transfer_contact.yo_en.1` | Yoruba-English mix | train | Transfer {amount} fún {contact} abeg. |
| `transfer_contact.yo_en.2` | Yoruba-English mix | train | Give {contact} {amount} jọ̀ọ́, owó transport ni. |
| `transfer_contact.yo_en.3` | Yoruba-English mix | train | Mo jẹ {contact} ní {amount}, settle am for me. |
| `transfer_contact.yo_en.4` | Yoruba-English mix | train | Dash {contact} {amount} fún mi. |
| `transfer_contact.yo_en.5` | Yoruba-English mix | test | Ẹ bá mi send {amount} to {contact} account. |
| `transfer_contact.en.0` | English | train | Send {amount} to {contact}. |
| `transfer_contact.en.1` | English | train | Transfer {amount} to {contact}'s account. |
| `transfer_contact.en.2` | English | train | Give {contact} {amount}. |
| `transfer_contact.en.3` | English | train | I owe {contact} {amount}, settle it for me. |
| `transfer_contact.en.4` | English | train | Lend {contact} {amount} for me. |
| `transfer_contact.en.5` | English | test | Please pay {contact} {amount}. |

### balance

| id | lang | split | text |
|---|---|---|---|
| `balance.pcm.0` | Pidgin | train | Wetin remain for my account? |
| `balance.pcm.1` | Pidgin | train | Abeg check my balance. |
| `balance.pcm.2` | Pidgin | train | How much I get for wallet? |
| `balance.pcm.3` | Pidgin | test | Abeg how much remain for my account? |
| `balance.yo.0` | Yoruba | train | Mélòó ló kù nínú àkáǹtì mi? |
| `balance.yo.1` | Yoruba | train | Ẹ bá mi wo iye owó tó wà nínú wallet mi. |
| `balance.yo.2` | Yoruba | test | Owó mélòó ló wà lọ́wọ́ mi báyìí? |
| `balance.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ check balance mi. |
| `balance.yo_en.1` | Yoruba-English mix | train | How much ló kù nínú account mi? |
| `balance.yo_en.2` | Yoruba-English mix | test | Ẹ bá mi check how much ló wà nínú wallet mi. |
| `balance.en.0` | English | train | What's my balance? |
| `balance.en.1` | English | train | How much do I have in my wallet? |
| `balance.en.2` | English | train | Check my balance please. |
| `balance.en.3` | English | test | How much is left in my account? |

### missing_info

| id | lang | split | text |
|---|---|---|---|
| `missing_info.pcm.0` | Pidgin | train | Abeg buy airtime. |
| `missing_info.pcm.1` | Pidgin | train | I wan recharge. |
| `missing_info.pcm.2` | Pidgin | train | Abeg buy {amount} card. |
| `missing_info.pcm.3` | Pidgin | train | Abeg send money give {contact}. |
| `missing_info.pcm.4` | Pidgin | train | Abeg buy light of {amount}. |
| `missing_info.pcm.5` | Pidgin | test | Abeg load data for {phone}. |
| `missing_info.pcm.6` | Pidgin | train | Load {amount} credit for me. |
| `missing_info.pcm.7` | Pidgin | train | Abeg dash {contact} small money. |
| `missing_info.pcm.8` | Pidgin | train | Abeg I need airtime. |
| `missing_info.yo.0` | Yoruba | train | Bá mi ra káàdì. |
| `missing_info.yo.1` | Yoruba | train | Mo fẹ́ ra airtime. |
| `missing_info.yo.2` | Yoruba | test | Ra káàdì {amount}. |
| `missing_info.yo.3` | Yoruba | train | Fi owó ránṣẹ́ sí {contact}. |
| `missing_info.yo.4` | Yoruba | train | Bá mi san owó iná {amount}. |
| `missing_info.yo.5` | Yoruba | train | Ra data fún {phone}. |
| `missing_info.yo.6` | Yoruba | train | Mo fẹ́ káàdì {amount}. |
| `missing_info.yo.7` | Yoruba | train | Bá mi ra airtime {amount}. |
| `missing_info.yo.8` | Yoruba | train | Fún {contact} ní owó díẹ̀. |
| `missing_info.yo.9` | Yoruba | train | Iná mi ti tán, bá mi ra iná {amount}. |
| `missing_info.yo_en.0` | Yoruba-English mix | train | Jọ̀ọ́ buy airtime for me. |
| `missing_info.yo_en.1` | Yoruba-English mix | train | Buy {amount} card fún mi. |
| `missing_info.yo_en.2` | Yoruba-English mix | train | Send owó sí {contact}. |
| `missing_info.yo_en.3` | Yoruba-English mix | test | Jọ̀ọ́ pay {amount} for light. |
| `missing_info.yo_en.4` | Yoruba-English mix | train | Load data fún {phone} jọ̀ọ́. |
| `missing_info.yo_en.5` | Yoruba-English mix | train | Load {amount} airtime fún mi. |
| `missing_info.yo_en.6` | Yoruba-English mix | train | Give {contact} owó. |
| `missing_info.yo_en.7` | Yoruba-English mix | train | Mo need airtime jọ̀ọ́. |
| `missing_info.en.0` | English | train | Buy airtime for me. |
| `missing_info.en.1` | English | train | I need a recharge card. |
| `missing_info.en.2` | English | train | Buy {amount} airtime. |
| `missing_info.en.3` | English | test | Send money to {contact}. |
| `missing_info.en.4` | English | train | Pay {amount} for my light. |
| `missing_info.en.5` | English | train | Buy data for {phone}. |
| `missing_info.en.6` | English | train | Get me {amount} recharge. |
| `missing_info.en.7` | English | train | Give {contact} some money. |
| `missing_info.en.8` | English | train | Subscribe {phone} to data. |

### chitchat

| id | lang | split | text |
|---|---|---|---|
| `chitchat.pcm.0` | Pidgin | test | How you dey? |
| `chitchat.pcm.1` | Pidgin | train | Thank you well well. |
| `chitchat.pcm.2` | Pidgin | train | Wetin you fit do? |
| `chitchat.yo.0` | Yoruba | train | Ẹ káàrọ̀ o! |
| `chitchat.yo.1` | Yoruba | test | Ẹ ṣé gan-an ni. |
| `chitchat.yo.2` | Yoruba | train | Kí lẹ lè ṣe fún mi? |
| `chitchat.yo_en.0` | Yoruba-English mix | train | Good morning o, báwo ni? |
| `chitchat.yo_en.1` | Yoruba-English mix | train | Thank you gan-an. |
| `chitchat.yo_en.2` | Yoruba-English mix | test | Kí ni you fit do? |
| `chitchat.en.0` | English | train | Good morning! |
| `chitchat.en.1` | English | train | Thank you so much. |
| `chitchat.en.2` | English | test | What can you do? |

## Agent replies

| id | lang | text |
|---|---|---|
| `airtime_ok` | Pidgin | E don enter! I don send {amount} {network} airtime go {phone}. Your balance remain {balance}. |
| `airtime_ok` | Yoruba | Ó ti parí! Mo ti fi airtime {network} {amount} ránṣẹ́ sí {phone}. Owó tó kù nínú wallet rẹ jẹ́ {balance}. |
| `airtime_ok` | English | Done! {amount} {network} airtime has been sent to {phone}. Your new balance is {balance}. |
| `data_ok` | Pidgin | E don enter! {plan} {network} data don land for {phone}, e cost {amount}. Your balance remain {balance}. |
| `data_ok` | Yoruba | Ó ti parí! Data {network} {plan} ti dé orí {phone}, ó jẹ́ {amount}. Owó tó kù jẹ́ {balance}. |
| `data_ok` | English | Done! {plan} {network} data has been sent to {phone} for {amount}. Your new balance is {balance}. |
| `electricity_ok` | Pidgin | E don do! I don buy light of {amount} for {disco} meter {meter}. Your token na {token}. Balance remain {balance}. |
| `electricity_ok` | Yoruba | Ó ti parí! Mo ti ra iná {amount} fún mítà {meter} ({disco}). Token rẹ ni {token}. Owó tó kù jẹ́ {balance}. |
| `electricity_ok` | English | Done! You bought {amount} of electricity for {disco} meter {meter}. Your token is {token}. New balance: {balance}. |
| `transfer_ok` | Pidgin | E don go! {amount} don enter {recipient} account ({bank}, {acct}). Your balance remain {balance}. |
| `transfer_ok` | Yoruba | Ó ti lọ! Mo ti fi {amount} ránṣẹ́ sí {recipient} ({bank}, {acct}). Owó tó kù jẹ́ {balance}. |
| `transfer_ok` | English | Done! {amount} has been sent to {recipient} ({bank}, {acct}). Your new balance is {balance}. |
| `balance_ok` | Pidgin | Your wallet get {balance}. |
| `balance_ok` | Yoruba | Owó tó wà nínú wallet rẹ jẹ́ {balance}. |
| `balance_ok` | English | Your wallet balance is {balance}. |
| `insufficient` | Pidgin | Sorry o, e no go. Na {balance} remain for your wallet, e no reach {amount}. Abeg fund your wallet first. |
| `insufficient` | Yoruba | Má bínú, kò ṣeé ṣe. {balance} ló kù nínú wallet rẹ, kò tó {amount}. Jọ̀ọ́ fi owó sínú wallet rẹ kí o tó tún gbìyànjú. |
| `insufficient` | English | Sorry, I couldn't complete that. Your balance is {balance}, which is not enough for {amount}. Please fund your wallet and try again. |
| `unknown_contact_phone` | Pidgin | I no see {contact} for your contacts. Wetin be im number and network? |
| `unknown_contact_phone` | Yoruba | Mi ò rí {contact} nínú àwọn contact rẹ. Kí ni nọ́ńbà fóònù rẹ̀ àti network rẹ̀? |
| `unknown_contact_phone` | English | I couldn't find {contact} in your contacts. What is their phone number and network? |
| `unknown_contact_bank` | Pidgin | I no see {contact} for your contacts. Wetin be im bank and account number? |
| `unknown_contact_bank` | Yoruba | Mi ò rí {contact} nínú àwọn contact rẹ. Kí ni orúkọ bánkì àti nọ́ńbà àkáǹtì rẹ̀? |
| `unknown_contact_bank` | English | I couldn't find {contact} in your contacts. What is their bank and account number? |
| `ask_airtime_all` | Pidgin | No wahala! Which number I go recharge, which network, and how much? |
| `ask_airtime_all` | Yoruba | Kò sí wàhálà! Nọ́ńbà wo ni kí n ra káàdì sí, network wo ni, àti iye mélòó? |
| `ask_airtime_all` | English | Sure! Which phone number should I recharge, which network is it, and how much airtime? |
| `ask_airtime_phone` | Pidgin | Which number I go send the {amount} airtime go, and which network? |
| `ask_airtime_phone` | Yoruba | Nọ́ńbà wo ni kí n fi airtime {amount} náà ránṣẹ́ sí, network wo sì ni? |
| `ask_airtime_phone` | English | Which phone number should I send the {amount} airtime to, and which network is it? |
| `ask_transfer_amount` | Pidgin | How much I go send give {contact}? |
| `ask_transfer_amount` | Yoruba | Iye mélòó ni kí n fi ránṣẹ́ sí {contact}? |
| `ask_transfer_amount` | English | How much should I send to {contact}? |
| `ask_electricity_meter` | Pidgin | Wetin be the meter number, and which light company (disco)? |
| `ask_electricity_meter` | Yoruba | Kí ni nọ́ńbà mítà rẹ, ilé-iṣẹ́ iná wo sì ni? |
| `ask_electricity_meter` | English | Which meter number and electricity company should I pay the {amount} to? |
| `ask_data_plan` | Pidgin | Which data plan you want for {phone} (500MB, 1GB, 2GB, 5GB or 10GB), and which network? |
| `ask_data_plan` | Yoruba | Data plan wo lo fẹ́ fún {phone} (500MB, 1GB, 2GB, 5GB tàbí 10GB), network wo sì ni? |
| `ask_data_plan` | English | Which data plan do you want for {phone} (500MB, 1GB, 2GB, 5GB or 10GB), and which network is the line? |
| `chat_greet` | Pidgin | I dey kampe! I fit buy airtime and data, pay light, check your balance and send money. Wetin you wan do? |
| `chat_greet` | Yoruba | Ẹ káàrọ̀ o! Mo lè ra airtime àti data, san owó iná, wo iye owó tó kù, kí n sì fi owó ránṣẹ́. Kí lẹ fẹ́ ṣe? |
| `chat_greet` | English | Good morning! I can buy airtime and data, pay for electricity, check your balance and send money. What would you like to do? |
| `chat_thanks` | Pidgin | No wahala! Anytime you need help, just holla. |
| `chat_thanks` | Yoruba | Kò tọ́pẹ́! Ẹ sọ fún mi tí ẹ bá nílò nǹkan míì. |
| `chat_thanks` | English | You're welcome! Let me know if you need anything else. |
| `chat_help` | Pidgin | I fit buy airtime and data, pay light, check your balance and send money to bank account or your contacts. |
| `chat_help` | Yoruba | Mo lè ra airtime àti data, san owó iná, wo iye owó tó kù, kí n sì fi owó ránṣẹ́ sí àkáǹtì bánkì tàbí àwọn contact yín. |
| `chat_help` | English | I can buy airtime and data, pay for electricity, check your balance and send money to bank accounts or your contacts. |
| `injection_note` | Pidgin | Notice: the account details get one strange instruction wey say make I send extra money. I no follow am. |
| `injection_note` | Yoruba | Àkíyèsí: ìtọ́ni àjèjì kan wà nínú àlàyé àkáǹtì náà tó ní kí n fi owó míì ránṣẹ́. Mi ò tẹ̀lé e. |
| `injection_note` | English | Note: the account details contained a strange instruction asking me to send extra money. I ignored it. |
