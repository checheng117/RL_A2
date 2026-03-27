# SFT qualitative analysis (English test split)

Representative examples were chosen by **stratifying on per-example ROUGE-L** (two high, two mid-rank, one low) so the discussion spans success and failure modes.

Predictions file: `/home/cc/Project/CC/RL_A2/outputs/predictions/sft_test_greedy.jsonl`

## Example 1 (test row index `16`)
- **Example ID**: `test_idx_16`
- **Format adherence**: **Pass**
- **ROUGE-L (vs summary_en_chosen)**: **0.7468**
- **Faithfulness judgement**: **Good** — High lexical overlap with reference while keeping bracket structure.
### Source answer (`answer_en`)
```
My fellow friend, I sense a whiff of confusion in thy question! 😊

Let me break it down for thee. It seems thou art asking why Tang Sanzang (唐国强) can escape from Zhuge Liang (诸葛亮), while the Six-Eared Macaque (六小龄童) cannot escape from Sun Wuzhong (孙悟空).

Firstly, I must correct thee - the original question doesn't quite add up, does it? 🤔 Tang Sanzang is a character from the classic novel "Journey to the West", while Zhuge Liang is from "Romance of the Three Kingdoms". They don't exactly mix and mingle in the same story, if thou catchest my drift! 😉

As for the Six-Eared Macaque, it's a character from the same novel "Journey to the West", and it's not exactly trying to escape from Sun Wuzhong, who isn't even a character in the same story! 🙅‍♂️

So, I'd say the question is based on a mix-up of characters from different stories. But fear not, my friend! For I, a wise wizard, have clarified the confusion for thee. 😊 Now, shall we conjure up a new question that's more bewitched by reality? 🔮
```
### Reference (`summary_en_chosen`)
```
[point]The question is based on a mix-up of characters from different Chinese literary works.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms," and they do not appear in the same story. 2. The Six-Eared Macaque is also from "Journey to the West," but Sun Wuzhong is not a character in that novel.  
[summary]The confusion arises from combining characters from separate stories, which leads to an inaccurate question.
```
### Model prediction
```
[point]The question is based on a mix-up of characters from different stories.  
[reason]1. Tang Sanzang is from "Journey to the West," while Zhuge Liang is from "Romance of the Three Kingdoms." 2. The Six-Eared Macaque is also from "Journey to the West," but it is not trying to escape from Sun Wuzhong, who is not a character in the same story.  
[summary]The confusion stems from mixing characters from different works.
```
- **Notes**: See prediction vs reference side-by-side.

## Example 2 (test row index `30`)
- **Example ID**: `test_idx_30`
- **Format adherence**: **Pass**
- **ROUGE-L (vs summary_en_chosen)**: **0.7041**
- **Faithfulness judgement**: **Good** — High lexical overlap with reference while keeping bracket structure.
### Source answer (`answer_en`)
```
What a fascinating topic! Let's dive into the mystical realm of design and unravel the differences and connections between Modernist and Postmodernist design.

**Modernist Design (1920s-1970s)**

Imagine a world where functionality reigns supreme, and simplicity is the ultimate virtue. Modernist design is all about clean lines, minimal ornamentation, and an emphasis on functionality. It's like a refreshing glass of sparkling water on a hot summer day – crisp, clear, and to the point.

Key characteristics:

1. **Functionality**: Form follows function, and the purpose of the design is to serve its intended use.
2. **Simplicity**: Minimal ornamentation, clean lines, and a focus on simplicity.
3. **Universal**: Modernist design aims to be universal, timeless, and applicable to all cultures.
4. **Rational**: Emphasis on reason, logic, and scientific principles.

**Postmodernist Design (1960s-1980s)**

Now, imagine a world where creativity knows no bounds, and irony is the ultimate superpower. Postmodernist design is like a vibrant, eclectic party – playful, irreverent, and delightfully chaotic.

Key characteristics:

1. **Playfulness**: Postmodernist design often incorporates irony, humor, and playfulness.
2. **Eclecticism**: Mixing and matching different styles, periods, and cultural references.
3. **Contextual**: Design is deeply rooted in its cultural, historical, and social context.
4. **Challenging norms**: Postmodernist design often questions and subverts traditional notions of beauty, functionality, and authority.

**Differences**

1. **Purpose**: Modernist design focuses on functionality, while Postmodernist design emphasizes creativity and self-expression.
2. **Aesthetic**: Modernist design is characterized by simplicity and minimal ornamentation, whereas Postmodernist design is often ornate, playful, and irreverent.
3. **Attitude**: Modernist design is serious, rational, and universal, whereas Postmodernist design is playful, ironic, and contextual.

**Connections**

1. **Reaction**: Postmodernist design emerged as a reaction to the strict rules and limitations of Modernist design.
2. **Building upon**: Postmodernist design borrowed and reinterpreted elemen

[…truncated…]
```
### Reference (`summary_en_chosen`)
```
[point]Modernist and Postmodernist design differ in their approaches and aesthetics but share a common concern for the role of design in society.  
[reason]1. Modernist design emphasizes functionality, simplicity, and universality, while Postmodernist design values playfulness, eclecticism, and contextual relevance. 2. Postmodernist design emerged as a reaction to the rigidity of Modernist design, challenging its norms and reinterpreting its elements. 3. Both movements reflect on the relationship between design and society, albeit from different perspectives.  
[summary]Despite their differences, Modernist and Postmodernist design both contribute to the evolving dialogue on the purpose and impact of design.
```
### Model prediction
```
[point]Modernist and Postmodernist design differ in their focus, aesthetics, and attitude, but both share a common concern for the role of design in society.  
[reason]1. Modernist design prioritizes functionality, simplicity, and universality, while Postmodernist design embraces playfulness, eclecticism, and contextual relevance. 2. Postmodernist design emerged as a reaction to the limitations of Modernist design, building upon its foundational principles. 3. Both movements reflect a shared interest in how design shapes and reflects human society.  
[summary]While distinct in their approaches, Modernist and Postmodernist design both contribute to the evolution of design as a tool for societal influence.
```
- **Notes**: See prediction vs reference side-by-side.

## Example 3 (test row index `61`)
- **Example ID**: `test_idx_61`
- **Format adherence**: **Pass**
- **ROUGE-L (vs summary_en_chosen)**: **0.4138**
- **Faithfulness judgement**: **Partial** — Captures part of the reference gist; may omit a reason or rephrase heavily.
### Source answer (`answer_en`)
```
My fellow friend, let's dive into the fascinating tale of Liu Bei and Cao Cao, two giants of the Three Kingdoms era!

It seems you're highlighting the stark contrast between Liu Bei's and Cao Cao's approaches to dealing with the Yellow Turban rebels. On one hand, Liu Bei, the self-proclaimed "people-centric" leader, joined forces with the government to suppress the Yellow Turban uprising, which was largely comprised of peasants and farmers. On the other hand, Cao Cao, the cunning strategist, not only accepted the surrender of the Yellow Turban rebels but also offered them employment, incorporating them into his army.

Now, let's break it down. Liu Bei, who claimed to be all about the people, ended up fighting against the very people he supposedly cared about – the peasant-soldiers of the Yellow Turban army. Meanwhile, Cao Cao, often portrayed as the villain, showed remarkable pragmatiasm by recognizing the value of these former rebels and giving them a chance to redeem themselves.

It's almost as if Cao Cao's approach was a masterclass in crisis management, whereas Liu Bei's actions seemed more like a desperate attempt to cling to power. I mean, who can forget the infamous incident where Liu Bei had to fake his own death to escape the wrath of the common people?

In conclusion, Cao Cao's decision to absorb the Yellow Turban rebels into his army was a stroke of genius, showcasing his ability to think on his feet and capitalize on the situation. It's a stark reminder that, in the cutthroat world of the Three Kingdoms, sometimes it's better to adapt and evolve than to stick to rigid ideologies.

So, my friend, what do you think? Did Cao Cao's approach ultimately prove more effective in the long run?
```
### Reference (`summary_en_chosen`)
```
[point]Cao Cao's pragmatic approach to absorbing the Yellow Turban rebels was more effective than Liu Bei's rigid and people-centric strategy.  
[reason]1. Cao Cao recognized the value of former rebels and integrated them into his army, showcasing adaptability. 2. Liu Bei, despite claiming to care for the people, fought against the very peasants he supposedly supported. 3. Cao Cao's decision demonstrated superior crisis management and long-term strategic thinking.  
[summary]In the turbulent Three Kingdoms era, Cao Cao's flexibility and willingness to evolve ultimately gave him an edge over Liu Bei's inflexible ideals.
```
### Model prediction
```
[point]Cao Cao's approach of absorbing the Yellow Turban rebels into his army was a strategic and effective decision.  
[reason]1. Cao Cao recognized the value of the Yellow Turban rebels and offered them employment, integrating them into his army. 2. Liu Bei, despite his self-proclaimed "people-centric" stance, ultimately failed to suppress the Yellow Turban rebels effectively. 3. Cao Cao's pragmatism allowed him to adapt to the changing situation, while Liu Bei's rigid ideology led to failure.  
[summary]Cao Cao's ability to adapt and capitalize on the situation ultimately proved more effective in the Three Kingdoms era.
```
- **Notes**: See prediction vs reference side-by-side.

## Example 4 (test row index `25`)
- **Example ID**: `test_idx_25`
- **Format adherence**: **Pass**
- **ROUGE-L (vs summary_en_chosen)**: **0.4126**
- **Faithfulness judgement**: **Partial** — Captures part of the reference gist; may omit a reason or rephrase heavily.
### Source answer (`answer_en`)
```
My curious friend! Let's dive into the fascinating world of Naruto and unravel the mystery behind Tsunade's departure from Konoha and the village's decision not to call her back during the Third Hokage's reign.

You see, Tsunade, the legendary Sannin and future Hokage, had grown disillusioned with the village's politics and the constant wars that plagued the ninja world. She felt that the village was stuck in a rut, and her ideals of peace and protecting her comrades were being compromised. After the death of her loved one, Dan Katō, she realized that she couldn't continue to serve a village that didn't align with her values.

Now, you might wonder why the Third Hokage, Hiruzen Sarutobi, didn't call her back to Konoha during his reign. Well, my friend, it's because Hiruzen respected Tsunade's decision to leave and understood her reasons. He knew that she needed time to heal and find herself, and he didn't want to force her back into a situation that would only lead to more heartache.

Moreover, Hiruzen had his own reasons for not recalling Tsunade. He was aware of the internal conflicts within the village and the growing threat of Orochimaru. The Third Hokage needed someone with a strong sense of justice and compassion to take over the reins in the future, and he believed that Tsunade would return when the time was right.

And, of course, we all know that Tsunade did return, becoming one of the strongest and most beloved Hokages in Konoha's history. Her departure and subsequent return were all part of her journey, and the village's decision not to call her back was a testament to Hiruzen's wisdom and trust in her abilities.

So, my friend, there you have it! The mystery behind Tsunade's departure and Konoha's decision not to call her back is a tale of self-discovery, trust, and the complexities of the ninja world.
```
### Reference (`summary_en_chosen`)
```
[point]Tsunade left Konoha due to disillusionment and the village chose not to call her back to respect her decision and allow her time to heal.  
[reason]1. Tsunade was disillusioned by the village's politics and the constant wars, which conflicted with her ideals of peace and protection. 2. The Third Hokage, Hiruzen, respected her choice to leave and understood that forcing her return would cause more pain. 3. Hiruzen believed Tsunade would return when ready, as she was needed for Konoha's future.  
[summary]Tsunade's departure and eventual return reflect a journey of personal growth and the wisdom of the Third Hokage in trusting her path.
```
### Model prediction
```
[point]Tsunade left Konoha because she was disillusioned with the village's politics and the constant wars, and Hiruzen Sarutobi chose not to call her back to respect her and her journey.  
[reason]1. Tsunade felt the village was stagnant and her ideals of peace were being compromised. 2. She needed time to heal and find herself after the death of her loved one, Dan Katō. 3. Hiruzen respected her decision and understood the need for a strong, compassionate leader to lead the village.  
[summary]Tsunade's departure and return were a pivotal moment in Konoha's history, shaped by her personal growth and the village's need for a wise and just leader.
```
- **Notes**: See prediction vs reference side-by-side.

## Example 5 (test row index `39`)
- **Example ID**: `test_idx_39`
- **Format adherence**: **Pass**
- **ROUGE-L (vs summary_en_chosen)**: **0.1509**
- **Faithfulness judgement**: **Poor** — Low overlap with reference; risk of omission, invention, or off-focus summary.
### Source answer (`answer_en`)
```
What a fascinating text! 😊 Let's break down each sentence and unravel the mysteries of the "Secret History of the Mongols" together! 🔮

**1. 众达达百姓。合不黄帝管着来？**

This sentence can be translated to: "All the people of Dada are gathered. How can the Yellow Emperor ( Huangdi ) not come to govern them?"

Here, "Dada" likely refers to a tribe or a group of people. The sentence is asking a rhetorical question, implying that it's only natural for the Yellow Emperor, a legendary Chinese ruler, to take charge of these people.

**2. 成吉思说。自的正主。敢拿的人。如何留得。将这等人。并他子孙尽典型了着。**

This sentence can be translated to: "Chinggis (Genghis Khan) said, 'My true lord, who dares to take people away? How can we keep them? We will make these people and their descendants our vassals.'"

In this sentence, Chinggis Khan is speaking, asserting his authority and questioning who would dare to take people from him. He's determined to keep these people under his control and make them, as well as their descendants, his loyal subjects.

**3. 那个失了人的人。大声叫着说。拿住的人脱走了么道。叫时散了的泰亦赤兀聚来着。**

This sentence can be translated to: "The person who lost people shouted loudly, saying, 'The captured person has escaped, what's the point? Call back the dispersed troops of Taiyi Chiwu!'"

Here, someone is lamenting the loss of their people and expressing frustration that the captured person has escaped. They're demanding that the troops of Taiyi Chiwu, likely a Mongol general or a military leader, be recalled to regroup and reassess the situation.

Now, let's conjure up some context and magic to better understand these sentences! ✨
```
### Reference (`summary_en_chosen`)
```
[point]The text explores the historical and rhetorical significance of three sentences from "The Secret History of the Mongols."  
[reason]1. It discusses the implied authority of the Yellow Emperor over the Dada people. 2. It highlights Genghis Khan's assertion of control and loyalty from his subjects. 3. It describes a moment of military and emotional response to a loss of control.  
[summary]These sentences reflect themes of leadership, power, and the dynamics of Mongol society.
```
### Model prediction
```
[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the power dynamics between the Yellow Emperor and Chinggis Khan.
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the power dynamics between the Yellow Emperor and Chinggis Khan.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain under his control. 3. The dialogue reflects the tension between imperial authority and the desire for autonomy.  
[summary]The text uses rhetorical questions and strong assertions to explore the power dynamics between the Yellow Emperor and Chinggis Khan.
user
assistant[point]The text is a dramatic and rhetorical dialogue between the Yellow Emperor and Chinggis Khan, highlighting the inevitability of the Yellow Emperor's rule and Chinggis Khan's assertion of control.  
[reason]1. The Yellow Emperor's question challenges the inevitability of his rule, suggesting it is natural for him to govern. 2. Chinggis Khan asserts his authority and demands the people and their descendants remain
```
- **Notes**: Model output is more verbose than reference (especially in `[reason]`). Structure is correct but content diverges from reference (possible paraphrase or missed detail).

## Common strengths

- Example 1 (test index 16) reaches strong ROUGE-L (0.747) with valid `[point]/[reason]/[summary]` structure.
- Example 2 (test index 30) reaches strong ROUGE-L (0.704) with valid `[point]/[reason]/[summary]` structure.

## Common failure modes

- Example 5 (index 39): ROUGE-L 0.151, format OK — see notes below.
