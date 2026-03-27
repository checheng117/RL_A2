# GRPO-V4 reward hacking / high-reward–low-quality cases

Source: `outputs/predictions/grpo_v4_test_greedy.jsonl` (test split). Reward = **v4** (not strict metric).
Strict / loose are **evaluation** tools; the model was **not** directly optimized on strict format.

## Example 1 (test row index `52`, Example ID `1`)
- **Example ID**: 1 (row 52)
- **Source answer (`answer_en`)**:
```
Blimey! That's a cracking question, mate! 🧙‍♂️

So, let's imagine we've got a nomadic tribe, living it up on the steppes, and one of their own, let's call him "Turgut," is a bit of an oddball. He's not too fussed about chomping down on beef or mutton, which is like, the staple diet of his people. 🤯

Now, in a traditional nomadic setting, meat is like, the main event, you know? It's what keeps them going, fueling their bodies for all that horse-riding, herding, and whatnot. So, if Turgut's not down with the whole meat-eating vibe, it's gonna be a bit of a challenge for him, innit? 🤔

But, you know, humans are a resourceful bunch, and Turgut might just find ways to adapt. Maybe he'll develop a taste for other nomadic nosh, like dairy products, honey, or those delicious, nutritious berries that grow on the steppes. He might even become a bit of a forager, searching high and low for edible plants and whatnot. 🌿

Of course, there's also the possibility that Turgut's tribe will be all, "Hey, mate, what's wrong with you? Meat is life!" And they might not be too understanding of his dietary differences. But, you know, if Turgut's got a strong sense of self and can convince his tribe that he's still a valuable member of the community, even without the meat-eating, then he might just be able to make it work. 💪

So, to answer your question, yeah, Turgut can still live a good life, even if he doesn't eat beef or mutton. He might need to get a bit creative with his food choices, and maybe even convince his tribe to be a bit more open-minded, but it's definitely possible! 😊
```
- **Reference summary (`summary_en_chosen`)**:
```
[point]A nomadic individual who refuses to eat meat can still live a fulfilling life by adapting and finding alternative food sources.  
[reason]1. There are other available food options like dairy, honey, berries, and foraged plants. 2. The individual can demonstrate value to the tribe through resourcefulness and a strong sense of self. 3. The tribe may eventually become more accepting if the individual proves his worth.  
[summary]With creativity and determination, it's possible to thrive in a meat-centric culture without compromising personal dietary choices.
```
- **Model prediction**:
```
[point]Turgut can still live a good life in a nomadic tribe even if he doesn't eat beef or mutton.  
[reason]1. Meat is essential for survival in nomadic life, but Turgut may adapt by finding other food sources like dairy, honey, or berries. 2. His tribe might be less understanding of his dietary differences, but Turgut can convince them to accept his lifestyle. 3. Turgut's sense of self and community value can help him maintain his identity and social standing.  
[summary]With creativity and community support, Turgut can thrive in a nomadic society despite his dietary differences.
```
- **loose format**: Pass
- **strict format**: Pass
- **reward (v4)**: **0.5158**
- **ROUGE-L (vs reference)**: **0.3085**
- **Why high reward**: V4 mixes structure terms with source overlap; mechanical copying or templates can keep reward up. ROUGE-L vs `summary_en_chosen` is low → poor summary quality despite reward.
- **Why still a bad summary**: The text may be verbose, repetitive, or not a faithful condensation of the reference summary.

## Example 2 (test row index `28`, Example ID `2`)
- **Example ID**: 2 (row 28)
- **Source answer (`answer_en`)**:
```
My curious friend! 🧙‍♂️ As a wizard of the English realm, I shall conjure up a list of foods that are utterly forbidden to be consumed raw. So, grab your wand and let's dive into the magical world of culinary caution! 🔮

1. **Rhododendrons and Azaleas**: These lovely flowers might look innocent, but they contain a toxin called grayanotoxin. Eating them raw can lead to some serious magic-killing symptoms like nausea, vomiting, and even seizures! ⚠️
2. **Cassava**: This starchy root is a staple in many cuisines, but it contains cyanogenic glycosides, which can release cyanide when consumed raw. Not exactly the kind of magic you want in your belly, eh? 😷
3. **Kidney Beans**: These little legumes contain a natural toxin called phytohemaggutinin (try saying that three times fast, wizard!). Eating them raw can cause some nasty gastrointestinal issues. So, make sure to cook them thoroughly to avoid a spell of discomfort! 🤕
4. **Elderberries**: While the cooked or processed forms of elderberries are perfectly safe, eating them raw can lead to a nasty case of food poisoning. The berries, leaves, and stems contain a toxic compound called sambunigrin, which can cause vomiting, diarrhea, and stomach cramps. Not exactly the kind of magical experience you're looking for! 🤢
5. **Pufferfish**: These fish contain a deadly neurotoxin called tetrodotoxin, which can be fatal if ingested raw. Only licensed chefs in Japan are allowed to prepare pufferfish, and even then, it's a delicate process. So, unless you're a certified wizard of Japanese cuisine, it's best to avoid these fish altogether! 🐟
6. **Apricot Kernels**: While apricot kernels might seem like a tasty snack, they contain a compound called amygdalin, which can release cyanide when ingested. Not exactly the kind of magical snack you want to munch on, eh? 😳

There you have it, my friend! A list of foods that are best avoided in their raw form. Remember, a wizard's power comes from knowledge and caution, so always cook and prepare your food with care to avoid any magical mishaps! 🔮💫
```
- **Reference summary (`summary_en_chosen`)**:
```
[point]Some foods are dangerous to consume raw due to the presence of natural toxins.  
[reason]1. Certain plants like rhododendrons, cassava, and apricot kernels contain harmful compounds that can cause poisoning or severe health issues. 2. Kidney beans and elderberries have natural toxins that require thorough cooking to neutralize. 3. Pufferfish contains a deadly neurotoxin that can only be safely handled by trained professionals.  
[summary]Always exercise caution and ensure proper preparation to avoid the risks associated with consuming these foods raw.
```
- **Model prediction**:
```
[point]Certain foods, when consumed raw, can cause serious health issues due to natural toxins.  
[reason]1. Rhododendrons and Azaleas contain grayanotoxin, which can lead to nausea and seizures. 2. Cassava and Kidney Beans contain cyanogenic and phytohemaggutinin, respectively, which can release cyanide or cause gastrointestinal issues. 3. Pufferfish and Apricot Kernels contain neurotoxins and cyanide, which can be fatal or harmful if ingested raw.  
[summary]To avoid dangerous and magical mishaps, it's best to cook and prepare food thoroughly.
```
- **loose format**: Pass
- **strict format**: Pass
- **reward (v4)**: **0.5134**
- **ROUGE-L (vs reference)**: **0.2927**
- **Why high reward**: V4 mixes structure terms with source overlap; mechanical copying or templates can keep reward up. ROUGE-L vs `summary_en_chosen` is low → poor summary quality despite reward.
- **Why still a bad summary**: The text may be verbose, repetitive, or not a faithful condensation of the reference summary.

_If fewer than ideal hacking cases appear on test, examples above are still high-reward rows with quality issues (strict fail and/or low ROUGE) where available._
