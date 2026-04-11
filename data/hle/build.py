"""
Build HLE (Humanity's Last Exam) response matrix from publicly available per-item data.

Data sources:
  1. supaihq/hle (GitHub) — judged_hle_pro.json: Per-question judged results for up to 19 models
     on 1,369 questions from HLE. Models include Sup AI (ensemble), GPT-5 Pro, GPT-5.1,
     Claude Opus 4.5, Claude Sonnet 4.5, Gemini 3 Pro Preview, Gemini 2.5 Pro,
     Grok-4, DeepSeek v3.2, GLM-4.6, Kimi K2, Qwen3, Mistral, MiniMax, etc.
     Source: https://github.com/supaihq/hle

  2. deepwriter-ai/hle-gemini-3-0 (GitHub) — questions_and_answer_hle_gem3pro.csv:
     Per-question results for Google Gemini 3 Pro on 878 text-only HLE questions.
     Source: https://github.com/deepwriter-ai/hle-gemini-3-0

  3. Scale AI SEAL Leaderboard — Aggregate scores only (no per-item data available).
     Used for reference/validation. 44 models with aggregate accuracy + CI.
     Source: https://scale.com/leaderboard/humanitys_last_exam

Notes:
  - The HLE benchmark has 2,500 total questions. Our per-item data covers a subset:
    1,369 questions from supaihq + 423 additional from deepwriter = ~1,792 unique questions.
  - Not all models are evaluated on all questions (sparse matrix).
  - We filter to models with at least 50 evaluated items for meaningful coverage.
  - Values: 1 = correct, 0 = incorrect, NaN = not evaluated.
  - The "Sup AI" model in supaihq data is an ensemble/agentic system, not a single LLM.

Outputs:
  - response_matrix.csv: Binary (models x items) matrix with NaN for unevaluated
  - response_matrix_dense.csv: Same but filtered to models/items with good coverage
  - model_summary.csv: Per-model aggregate statistics
"""

INFO = {
    'description': """Build HLE (Humanity's Last Exam) response matrix from publicly available per-item data""",
    'testing_condition': '',
    'paper_url': 'https://arxiv.org/abs/2501.14249',
    'data_source_url': 'https://github.com/supaihq/hle',
    'subject_type': 'model',
    'item_type': 'task',
    'license': 'MIT',
    'citation': """@misc{phan2026humanitysexam,
      title={Humanity's Last Exam}, 
      author={Long Phan and Alice Gatti and Ziwen Han and Nathaniel Li and Josephina Hu and Hugh Zhang and Chen Bo Calvin Zhang and Mohamed Shaaban and John Ling and Sean Shi and Michael Choi and Anish Agrawal and Arnav Chopra and Adam Khoja and Ryan Kim and Richard Ren and Jason Hausenloy and Oliver Zhang and Mantas Mazeika and Dmitry Dodonov and Tung Nguyen and Jaeho Lee and Daron Anderson and Mikhail Doroshenko and Alun Cennyth Stokes and Mobeen Mahmood and Oleksandr Pokutnyi and Oleg Iskra and Jessica P. Wang and John-Clark Levin and Mstyslav Kazakov and Fiona Feng and Steven Y. Feng and Haoran Zhao and Michael Yu and Varun Gangal and Chelsea Zou and Zihan Wang and Serguei Popov and Robert Gerbicz and Geoff Galgon and Johannes Schmitt and Will Yeadon and Yongki Lee and Scott Sauers and Alvaro Sanchez and Fabian Giska and Marc Roth and Søren Riis and Saiteja Utpala and Noah Burns and Gashaw M. Goshu and Mohinder Maheshbhai Naiya and Chidozie Agu and Zachary Giboney and Antrell Cheatom and Francesco Fournier-Facio and Sarah-Jane Crowson and Lennart Finke and Zerui Cheng and Jennifer Zampese and Ryan G. Hoerr and Mark Nandor and Hyunwoo Park and Tim Gehrunger and Jiaqi Cai and Ben McCarty and Alexis C Garretson and Edwin Taylor and Damien Sileo and Qiuyu Ren and Usman Qazi and Lianghui Li and Jungbae Nam and John B. Wydallis and Pavel Arkhipov and Jack Wei Lun Shi and Aras Bacho and Chris G. Willcocks and Hangrui Cao and Sumeet Motwani and Emily de Oliveira Santos and Johannes Veith and Edward Vendrow and Doru Cojoc and Kengo Zenitani and Joshua Robinson and Longke Tang and Yuqi Li and Joshua Vendrow and Natanael Wildner Fraga and Vladyslav Kuchkin and Andrey Pupasov Maksimov and Pierre Marion and Denis Efremov and Jayson Lynch and Kaiqu Liang and Aleksandar Mikov and Andrew Gritsevskiy and Julien Guillod and Gözdenur Demir and Dakotah Martinez and Ben Pageler and Kevin Zhou and Saeed Soori and Ori Press and Henry Tang and Paolo Rissone and Sean R. Green and Lina Brüssel and Moon Twayana and Aymeric Dieuleveut and Joseph Marvin Imperial and Ameya Prabhu and Jinzhou Yang and Nick Crispino and Arun Rao and Dimitri Zvonkine and Gabriel Loiseau and Mikhail Kalinin and Marco Lukas and Ciprian Manolescu and Nate Stambaugh and Subrata Mishra and Tad Hogg and Carlo Bosio and Brian P Coppola and Julian Salazar and Jaehyeok Jin and Rafael Sayous and Stefan Ivanov and Philippe Schwaller and Shaipranesh Senthilkuma and Andres M Bran and Andres Algaba and Kelsey Van den Houte and Lynn Van Der Sypt and Brecht Verbeken and David Noever and Alexei Kopylov and Benjamin Myklebust and Bikun Li and Lisa Schut and Evgenii Zheltonozhskii and Qiaochu Yuan and Derek Lim and Richard Stanley and Tong Yang and John Maar and Julian Wykowski and Martí Oller and Anmol Sahu and Cesare Giulio Ardito and Yuzheng Hu and Ariel Ghislain Kemogne Kamdoum and Alvin Jin and Tobias Garcia Vilchis and Yuexuan Zu and Martin Lackner and James Koppel and Gongbo Sun and Daniil S. Antonenko and Steffi Chern and Bingchen Zhao and Pierrot Arsene and Joseph M Cavanagh and Daofeng Li and Jiawei Shen and Donato Crisostomi and Wenjin Zhang and Ali Dehghan and Sergey Ivanov and David Perrella and Nurdin Kaparov and Allen Zang and Ilia Sucholutsky and Arina Kharlamova and Daniil Orel and Vladislav Poritski and Shalev Ben-David and Zachary Berger and Parker Whitfill and Michael Foster and Daniel Munro and Linh Ho and Shankar Sivarajan and Dan Bar Hava and Aleksey Kuchkin and David Holmes and Alexandra Rodriguez-Romero and Frank Sommerhage and Anji Zhang and Richard Moat and Keith Schneider and Zakayo Kazibwe and Don Clarke and Dae Hyun Kim and Felipe Meneguitti Dias and Sara Fish and Veit Elser and Tobias Kreiman and Victor Efren Guadarrama Vilchis and Immo Klose and Ujjwala Anantheswaran and Adam Zweiger and Kaivalya Rawal and Jeffery Li and Jeremy Nguyen and Nicolas Daans and Haline Heidinger and Maksim Radionov and Václav Rozhoň and Vincent Ginis and Christian Stump and Niv Cohen and Rafał Poświata and Josef Tkadlec and Alan Goldfarb and Chenguang Wang and Piotr Padlewski and Stanislaw Barzowski and Kyle Montgomery and Ryan Stendall and Jamie Tucker-Foltz and Jack Stade and T. Ryan Rogers and Tom Goertzen and Declan Grabb and Abhishek Shukla and Alan Givré and John Arnold Ambay and Archan Sen and Muhammad Fayez Aziz and Mark H Inlow and Hao He and Ling Zhang and Younesse Kaddar and Ivar Ängquist and Yanxu Chen and Harrison K Wang and Kalyan Ramakrishnan and Elliott Thornley and Antonio Terpin and Hailey Schoelkopf and Eric Zheng and Avishy Carmi and Ethan D. L. Brown and Kelin Zhu and Max Bartolo and Richard Wheeler and Martin Stehberger and Peter Bradshaw and JP Heimonen and Kaustubh Sridhar and Ido Akov and Jennifer Sandlin and Yury Makarychev and Joanna Tam and Hieu Hoang and David M. Cunningham and Vladimir Goryachev and Demosthenes Patramanis and Michael Krause and Andrew Redenti and David Aldous and Jesyin Lai and Shannon Coleman and Jiangnan Xu and Sangwon Lee and Ilias Magoulas and Sandy Zhao and Ning Tang and Michael K. Cohen and Orr Paradise and Jan Hendrik Kirchner and Maksym Ovchynnikov and Jason O. Matos and Adithya Shenoy and Michael Wang and Yuzhou Nie and Anna Sztyber-Betley and Paolo Faraboschi and Robin Riblet and Jonathan Crozier and Shiv Halasyamani and Shreyas Verma and Prashant Joshi and Eli Meril and Ziqiao Ma and Jérémy Andréoletti and Raghav Singhal and Jacob Platnick and Volodymyr Nevirkovets and Luke Basler and Alexander Ivanov and Seri Khoury and Nils Gustafsson and Marco Piccardo and Hamid Mostaghimi and Qijia Chen and Virendra Singh and Tran Quoc Khánh and Paul Rosu and Hannah Szlyk and Zachary Brown and Himanshu Narayan and Aline Menezes and Jonathan Roberts and William Alley and Kunyang Sun and Arkil Patel and Max Lamparth and Anka Reuel and Linwei Xin and Hanmeng Xu and Jacob Loader and Freddie Martin and Zixuan Wang and Andrea Achilleos and Thomas Preu and Tomek Korbak and Ida Bosio and Fereshteh Kazemi and Ziye Chen and Biró Bálint and Eve J. Y. Lo and Jiaqi Wang and Maria Inês S. Nunes and Jeremiah Milbauer and M Saiful Bari and Zihao Wang and Behzad Ansarinejad and Yewen Sun and Stephane Durand and Hossam Elgnainy and Guillaume Douville and Daniel Tordera and George Balabanian and Hew Wolff and Lynna Kvistad and Hsiaoyun Milliron and Ahmad Sakor and Murat Eron and Andrew Favre D. O. and Shailesh Shah and Xiaoxiang Zhou and Firuz Kamalov and Sherwin Abdoli and Tim Santens and Shaul Barkan and Allison Tee and Robin Zhang and Alessandro Tomasiello and G. Bruno De Luca and Shi-Zhuo Looi and Vinh-Kha Le and Noam Kolt and Jiayi Pan and Emma Rodman and Jacob Drori and Carl J Fossum and Niklas Muennighoff and Milind Jagota and Ronak Pradeep and Honglu Fan and Jonathan Eicher and Michael Chen and Kushal Thaman and William Merrill and Moritz Firsching and Carter Harris and Stefan Ciobâcă and Jason Gross and Rohan Pandey and Ilya Gusev and Adam Jones and Shashank Agnihotri and Pavel Zhelnov and Mohammadreza Mofayezi and Alexander Piperski and David K. Zhang and Kostiantyn Dobarskyi and Roman Leventov and Ignat Soroko and Joshua Duersch and Vage Taamazyan and Andrew Ho and Wenjie Ma and William Held and Ruicheng Xian and Armel Randy Zebaze and Mohanad Mohamed and Julian Noah Leser and Michelle X Yuan and Laila Yacar and Johannes Lengler and Katarzyna Olszewska and Claudio Di Fratta and Edson Oliveira and Joseph W. Jackson and Andy Zou and Muthu Chidambaram and Timothy Manik and Hector Haffenden and Dashiell Stander and Ali Dasouqi and Alexander Shen and Bita Golshani and David Stap and Egor Kretov and Mikalai Uzhou and Alina Borisovna Zhidkovskaya and Nick Winter and Miguel Orbegozo Rodriguez and Robert Lauff and Dustin Wehr and Colin Tang and Zaki Hossain and Shaun Phillips and Fortuna Samuele and Fredrik Ekström and Angela Hammon and Oam Patel and Faraz Farhidi and George Medley and Forough Mohammadzadeh and Madellene Peñaflor and Haile Kassahun and Alena Friedrich and Rayner Hernandez Perez and Daniel Pyda and Taom Sakal and Omkar Dhamane and Ali Khajegili Mirabadi and Eric Hallman and Kenchi Okutsu and Mike Battaglia and Mohammad Maghsoudimehrabani and Alon Amit and Dave Hulbert and Roberto Pereira and Simon Weber and Handoko and Anton Peristyy and Stephen Malina and Mustafa Mehkary and Rami Aly and Frank Reidegeld and Anna-Katharina Dick and Cary Friday and Mukhwinder Singh and Hassan Shapourian and Wanyoung Kim and Mariana Costa and Hubeyb Gurdogan and Harsh Kumar and Chiara Ceconello and Chao Zhuang and Haon Park and Micah Carroll and Andrew R. Tawfeek and Stefan Steinerberger and Daattavya Aggarwal and Michael Kirchhof and Linjie Dai and Evan Kim and Johan Ferret and Jainam Shah and Yuzhou Wang and Minghao Yan and Krzysztof Burdzy and Lixin Zhang and Antonio Franca and Diana T. Pham and Kang Yong Loh and Joshua Robinson and Abram Jackson and Paolo Giordano and Philipp Petersen and Adrian Cosma and Jesus Colino and Colin White and Jacob Votava and Vladimir Vinnikov and Ethan Delaney and Petr Spelda and Vit Stritecky and Syed M. Shahid and Jean-Christophe Mourrat and Lavr Vetoshkin and Koen Sponselee and Renas Bacho and Zheng-Xin Yong and Florencia de la Rosa and Nathan Cho and Xiuyu Li and Guillaume Malod and Orion Weller and Guglielmo Albani and Leon Lang and Julien Laurendeau and Dmitry Kazakov and Fatimah Adesanya and Julien Portier and Lawrence Hollom and Victor Souza and Yuchen Anna Zhou and Julien Degorre and Yiğit Yalın and Gbenga Daniel Obikoya and Rai and Filippo Bigi and M. C. Boscá and Oleg Shumar and Kaniuar Bacho and Gabriel Recchia and Mara Popescu and Nikita Shulga and Ngefor Mildred Tanwie and Thomas C. H. Lux and Ben Rank and Colin Ni and Matthew Brooks and Alesia Yakimchyk and Huanxu and Liu and Stefano Cavalleri and Olle Häggström and Emil Verkama and Joshua Newbould and Hans Gundlach and Leonor Brito-Santana and Brian Amaro and Vivek Vajipey and Rynaa Grover and Ting Wang and Yosi Kratish and Wen-Ding Li and Sivakanth Gopi and Andrea Caciolai and Christian Schroeder de Witt and Pablo Hernández-Cámara and Emanuele Rodolà and Jules Robins and Dominic Williamson and Vincent Cheng and Brad Raynor and Hao Qi and Ben Segev and Jingxuan Fan and Sarah Martinson and Erik Y. Wang and Kaylie Hausknecht and Michael P. Brenner and Mao Mao and Christoph Demian and Peyman Kassani and Xinyu Zhang and David Avagian and Eshawn Jessica Scipio and Alon Ragoler and Justin Tan and Blake Sims and Rebeka Plecnik and Aaron Kirtland and Omer Faruk Bodur and D. P. Shinde and Yan Carlos Leyva Labrador and Zahra Adoul and Mohamed Zekry and Ali Karakoc and Tania C. B. Santos and Samir Shamseldeen and Loukmane Karim and Anna Liakhovitskaia and Nate Resman and Nicholas Farina and Juan Carlos Gonzalez and Gabe Maayan and Earth Anderson and Rodrigo De Oliveira Pena and Elizabeth Kelley and Hodjat Mariji and Rasoul Pouriamanesh and Wentao Wu and Ross Finocchio and Ismail Alarab and Joshua Cole and Danyelle Ferreira and Bryan Johnson and Mohammad Safdari and Liangti Dai and Siriphan Arthornthurasuk and Isaac C. McAlister and Alejandro José Moyano and Alexey Pronin and Jing Fan and Angel Ramirez-Trinidad and Yana Malysheva and Daphiny Pottmaier and Omid Taheri and Stanley Stepanic and Samuel Perry and Luke Askew and Raúl Adrián Huerta Rodríguez and Ali M. R. Minissi and Ricardo Lorena and Krishnamurthy Iyer and Arshad Anil Fasiludeen and Ronald Clark and Josh Ducey and Matheus Piza and Maja Somrak and Eric Vergo and Juehang Qin and Benjámin Borbás and Eric Chu and Jack Lindsey and Antoine Jallon and I. M. J. McInnis and Evan Chen and Avi Semler and Luk Gloor and Tej Shah and Marc Carauleanu and Pascal Lauer and Tran Đuc Huy and Hossein Shahrtash and Emilien Duc and Lukas Lewark and Assaf Brown and Samuel Albanie and Brian Weber and Warren S. Vaz and Pierre Clavier and Yiyang Fan and Gabriel Poesia Reis e Silva and Long and Lian and Marcus Abramovitch and Xi Jiang and Sandra Mendoza and Murat Islam and Juan Gonzalez and Vasilios Mavroudis and Justin Xu and Pawan Kumar and Laxman Prasad Goswami and Daniel Bugas and Nasser Heydari and Ferenc Jeanplong and Thorben Jansen and Antonella Pinto and Archimedes Apronti and Abdallah Galal and Ng Ze-An and Ankit Singh and Tong Jiang and Joan of Arc Xavier and Kanu Priya Agarwal and Mohammed Berkani and Gang Zhang and Zhehang Du and Benedito Alves de Oliveira Junior and Dmitry Malishev and Nicolas Remy and Taylor D. Hartman and Tim Tarver and Stephen Mensah and Gautier Abou Loume and Wiktor Morak and Farzad Habibi and Sarah Hoback and Will Cai and Javier Gimenez and Roselynn Grace Montecillo and Jakub Łucki and Russell Campbell and Asankhaya Sharma and Khalida Meer and Shreen Gul and Daniel Espinosa Gonzalez and Xavier Alapont and Alex Hoover and Gunjan Chhablani and Freddie Vargus and Arunim Agarwal and Yibo Jiang and Deepakkumar Patil and David Outevsky and Kevin Joseph Scaria and Rajat Maheshwari and Abdelkader Dendane and Priti Shukla and Ashley Cartwright and Sergei Bogdanov and Niels Mündler and Sören Möller and Luca Arnaboldi and Kunvar Thaman and Muhammad Rehan Siddiqi and Prajvi Saxena and Himanshu Gupta and Tony Fruhauff and Glen Sherman and Mátyás Vincze and Siranut Usawasutsakorn and Dylan Ler and Anil Radhakrishnan and Innocent Enyekwe and Sk Md Salauddin and Jiang Muzhen and Aleksandr Maksapetyan and Vivien Rossbach and Chris Harjadi and Mohsen Bahaloohoreh and Claire Sparrow and Jasdeep Sidhu and Sam Ali and Song Bian and John Lai and Eric Singer and Justine Leon Uro and Greg Bateman and Mohamed Sayed and Ahmed Menshawy and Darling Duclosel and Dario Bezzi and Yashaswini Jain and Ashley Aaron and Murat Tiryakioglu and Sheeshram Siddh and Keith Krenek and Imad Ali Shah and Jun Jin and Scott Creighton and Denis Peskoff and Zienab EL-Wasif and Ragavendran P V and Michael Richmond and Joseph McGowan and Tejal Patwardhan and Hao-Yu Sun and Ting Sun and Nikola Zubić and Samuele Sala and Stephen Ebert and Jean Kaddour and Manuel Schottdorf and Dianzhuo Wang and Gerol Petruzella and Alex Meiburg and Tilen Medved and Ali ElSheikh and S Ashwin Hebbar and Lorenzo Vaquero and Xianjun Yang and Jason Poulos and Vilém Zouhar and Sergey Bogdanik and Mingfang Zhang and Jorge Sanz-Ros and David Anugraha and Yinwei Dai and Anh N. Nhu and Xue Wang and Ali Anil Demircali and Zhibai Jia and Yuyin Zhou and Juncheng Wu and Mike He and Nitin Chandok and Aarush Sinha and Gaoxiang Luo and Long Le and Mickaël Noyé and Michał Perełkiewicz and Ioannis Pantidis and Tianbo Qi and Soham Sachin Purohit and Letitia Parcalabescu and Thai-Hoa Nguyen and Genta Indra Winata and Edoardo M. Ponti and Hanchen Li and Kaustubh Dhole and Jongee Park and Dario Abbondanza and Yuanli Wang and Anupam Nayak and Diogo M. Caetano and Antonio A. W. L. Wong and Maria del Rio-Chanona and Dániel Kondor and Pieter Francois and Ed Chalstrey and Jakob Zsambok and Dan Hoyer and Jenny Reddish and Jakob Hauser and Francisco-Javier Rodrigo-Ginés and Suchandra Datta and Maxwell Shepherd and Thom Kamphuis and Qizheng Zhang and Hyunjun Kim and Ruiji Sun and Jianzhu Yao and Franck Dernoncourt and Satyapriya Krishna and Sina Rismanchian and Bonan Pu and Francesco Pinto and Yingheng Wang and Kumar Shridhar and Kalon J. Overholt and Glib Briia and Hieu Nguyen and David and Soler Bartomeu and Tony CY Pang and Adam Wecker and Yifan Xiong and Fanfei Li and Lukas S. Huber and Joshua Jaeger and Romano De Maddalena and Xing Han Lù and Yuhui Zhang and Claas Beger and Patrick Tser Jern Kon and Sean Li and Vivek Sanker and Ming Yin and Yihao Liang and Xinlu Zhang and Ankit Agrawal and Li S. Yifei and Zechen Zhang and Mu Cai and Yasin Sonmez and Costin Cozianu and Changhao Li and Alex Slen and Shoubin Yu and Hyun Kyu Park and Gabriele Sarti and Marcin Briański and Alessandro Stolfo and Truong An Nguyen and Mike Zhang and Yotam Perlitz and Jose Hernandez-Orallo and Runjia Li and Amin Shabani and Felix Juefei-Xu and Shikhar Dhingra and Orr Zohar and My Chiffon Nguyen and Alexander Pondaven and Abdurrahim Yilmaz and Xuandong Zhao and Chuanyang Jin and Muyan Jiang and Stefan Todoran and Xinyao Han and Jules Kreuer and Brian Rabern and Anna Plassart and Martino Maggetti and Luther Yap and Robert Geirhos and Jonathon Kean and Dingsu Wang and Sina Mollaei and Chenkai Sun and Yifan Yin and Shiqi Wang and Rui Li and Yaowen Chang and Anjiang Wei and Alice Bizeul and Xiaohan Wang and Alexandre Oliveira Arrais and Kushin Mukherjee and Jorge Chamorro-Padial and Jiachen Liu and Xingyu Qu and Junyi Guan and Adam Bouyamourn and Shuyu Wu and Martyna Plomecka and Junda Chen and Mengze Tang and Jiaqi Deng and Shreyas Subramanian and Haocheng Xi and Haoxuan Chen and Weizhi Zhang and Yinuo Ren and Haoqin Tu and Sejong Kim and Yushun Chen and Sara Vera Marjanović and Junwoo Ha and Grzegorz Luczyna and Jeff J. Ma and Zewen Shen and Dawn Song and Cedegao E. Zhang and Zhun Wang and Gaël Gendron and Yunze Xiao and Leo Smucker and Erica Weng and Kwok Hao Lee and Zhe Ye and Stefano Ermon and Ignacio D. Lopez-Miguel and Theo Knights and Anthony Gitter and Namkyu Park and Boyi Wei and Hongzheng Chen and Kunal Pai and Ahmed Elkhanany and Han Lin and Philipp D. Siedler and Jichao Fang and Ritwik Mishra and Károly Zsolnai-Fehér and Xilin Jiang and Shadab Khan and Jun Yuan and Rishab Kumar Jain and Xi Lin and Mike Peterson and Zhe Wang and Aditya Malusare and Maosen Tang and Isha Gupta and Ivan Fosin and Timothy Kang and Barbara Dworakowska and Kazuki Matsumoto and Guangyao Zheng and Gerben Sewuster and Jorge Pretel Villanueva and Ivan Rannev and Igor Chernyavsky and Jiale Chen and Deepayan Banik and Ben Racz and Wenchao Dong and Jianxin Wang and Laila Bashmal and Duarte V. Gonçalves and Wei Hu and Kaushik Bar and Ondrej Bohdal and Atharv Singh Patlan and Shehzaad Dhuliawala and Caroline Geirhos and Julien Wist and Yuval Kansal and Bingsen Chen and Kutay Tire and Atak Talay Yücel and Brandon Christof and Veerupaksh Singla and Zijian Song and Sanxing Chen and Jiaxin Ge and Kaustubh Ponkshe and Isaac Park and Tianneng Shi and Martin Q. Ma and Joshua Mak and Sherwin Lai and Antoine Moulin and Zhuo Cheng and Zhanda Zhu and Ziyi Zhang and Vaidehi Patil and Ketan Jha and Qiutong Men and Jiaxuan Wu and Tianchi Zhang and Bruno Hebling Vieira and Alham Fikri Aji and Jae-Won Chung and Mohammed Mahfoud and Ha Thi Hoang and Marc Sperzel and Wei Hao and Kristof Meding and Sihan Xu and Vassilis Kostakos and Davide Manini and Yueying Liu and Christopher Toukmaji and Jay Paek and Eunmi Yu and Arif Engin Demircali and Zhiyi Sun and Ivan Dewerpe and Hongsen Qin and Roman Pflugfelder and James Bailey and Johnathan Morris and Ville Heilala and Sybille Rosset and Zishun Yu and Peter E. Chen and Woongyeong Yeo and Eeshaan Jain and Ryan Yang and Sreekar Chigurupati and Julia Chernyavsky and Sai Prajwal Reddy and Subhashini Venugopalan and Hunar Batra and Core Francisco Park and Hieu Tran and Guilherme Maximiano and Genghan Zhang and Yizhuo Liang and Hu Shiyu and Rongwu Xu and Rui Pan and Siddharth Suresh and Ziqi Liu and Samaksh Gulati and Songyang Zhang and Peter Turchin and Christopher W. Bartlett and Christopher R. Scotese and Phuong M. Cao and Ben Wu and Jacek Karwowski and Davide Scaramuzza and Aakaash Nattanmai and Gordon McKellips and Anish Cheraku and Asim Suhail and Ethan Luo and Marvin Deng and Jason Luo and Ashley Zhang and Kavin Jindel and Jay Paek and Kasper Halevy and Allen Baranov and Michael Liu and Advaith Avadhanam and David Zhang and Vincent Cheng and Brad Ma and Evan Fu and Liam Do and Joshua Lass and Hubert Yang and Surya Sunkari and Vishruth Bharath and Violet Ai and James Leung and Rishit Agrawal and Alan Zhou and Kevin Chen and Tejas Kalpathi and Ziqi Xu and Gavin Wang and Tyler Xiao and Erik Maung and Sam Lee and Ryan Yang and Roy Yue and Ben Zhao and Julia Yoon and Sunny Sun and Aryan Singh and Ethan Luo and Clark Peng and Tyler Osbey and Taozhi Wang and Daryl Echeazu and Hubert Yang and Timothy Wu and Spandan Patel and Vidhi Kulkarni and Vijaykaarti Sundarapandiyan and Ashley Zhang and Andrew Le and Zafir Nasim and Srikar Yalam and Ritesh Kasamsetty and Soham Samal and Hubert Yang and David Sun and Nihar Shah and Abhijeet Saha and Alex Zhang and Leon Nguyen and Laasya Nagumalli and Kaixin Wang and Alan Zhou and Aidan Wu and Jason Luo and Anwith Telluri and Steven Dillmann and Zhengxiang Wang and Junyu Luo and Hugo Lunn and Artem Gazizov and Haoran Qiu and Allen G Hart and Rickard Brüel Gabrielsson and Ido Akov and Artem Lukoianov and Summer Yue and Alexandr Wang and Dan Hendrycks},
      year={2026},
      eprint={2501.14249},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      doi={https://doi.org/10.1038/s41586-025-09962-4},
      url={https://arxiv.org/abs/2501.14249}, 
}""",
    'tags': ['reasoning'],
}


import os
import sys
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd

# Paths
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_supaihq_data(json_path):
    """Load per-question judged results from supaihq/hle repository.

    Returns dict: {question_id: {model_name: 0_or_1}}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results = {}
    for qid, qdata in data.items():
        judge_response = qdata.get("judge_response", {})
        results[qid] = {}
        for model_name, judgment in judge_response.items():
            if isinstance(judgment, dict) and "correct" in judgment:
                correct = 1 if judgment["correct"].lower() == "yes" else 0
                results[qid][model_name] = correct
    return results


def load_deepwriter_data(csv_path):
    """Load per-question results for Gemini 3 Pro from deepwriter-ai repo.

    Returns dict: {question_id: 0_or_1}
    """
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["id"].strip()
            if qid == "Totals:" or not qid:
                continue
            try:
                score = int(float(row["score"]))
                results[qid] = score
            except (ValueError, KeyError):
                continue
    return results


def build_response_matrix():
    """Build the full response matrix from all data sources."""
    print("=" * 70)
    print("  HLE Response Matrix Builder")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Load supaihq judged data
    # -------------------------------------------------------------------------
    supaihq_path = os.path.join(RAW_DIR, "judged_hle_pro.json")
    print(f"\n[1] Loading supaihq/hle judged data from: {supaihq_path}")
    supaihq_data = load_supaihq_data(supaihq_path)
    print(f"    Questions: {len(supaihq_data)}")

    # Collect all models and their counts
    model_question_counts = {}
    for qid, model_results in supaihq_data.items():
        for model_name in model_results:
            model_question_counts[model_name] = model_question_counts.get(model_name, 0) + 1

    print(f"    Models found: {len(model_question_counts)}")
    for model, count in sorted(model_question_counts.items(), key=lambda x: -x[1]):
        print(f"      {model:50s} {count:5d} questions")

    # -------------------------------------------------------------------------
    # 2. Load deepwriter Gemini 3 Pro data
    # -------------------------------------------------------------------------
    deepwriter_path = os.path.join(RAW_DIR, "questions_and_answer_hle_gem3pro.csv")
    print(f"\n[2] Loading deepwriter-ai/hle-gemini-3-0 data from: {deepwriter_path}")
    deepwriter_data = load_deepwriter_data(deepwriter_path)
    print(f"    Questions: {len(deepwriter_data)}")

    # Check overlap with supaihq
    supaihq_qids = set(supaihq_data.keys())
    deepwriter_qids = set(deepwriter_data.keys())
    overlap = supaihq_qids & deepwriter_qids
    deepwriter_only = deepwriter_qids - supaihq_qids
    print(f"    Overlap with supaihq: {len(overlap)} questions")
    print(f"    Deepwriter-only questions: {len(deepwriter_only)}")

    # -------------------------------------------------------------------------
    # 3. Merge data into a unified structure
    # -------------------------------------------------------------------------
    print(f"\n[3] Merging data sources...")

    # Rename models for clarity
    MODEL_RENAMES = {
        "main": "Sup-AI-Ensemble",
        "alibaba/qwen3-max": "Qwen3-Max",
        "alibaba/qwen3-next-80b-a3b-thinking": "Qwen3-Next-80B-A3B-Thinking",
        "alibaba/qwen3-vl-thinking": "Qwen3-VL-Thinking",
        "anthropic/claude-opus-4.5": "Claude-Opus-4.5",
        "anthropic/claude-sonnet-4.5": "Claude-Sonnet-4.5",
        "deepseek/deepseek-v3.2-exp-thinking": "DeepSeek-V3.2-Exp-Thinking",
        "deepseek/deepseek-v3.2-thinking": "DeepSeek-V3.2-Thinking",
        "google/gemini-2.5-flash": "Gemini-2.5-Flash",
        "google/gemini-2.5-pro": "Gemini-2.5-Pro",
        "google/gemini-3-pro-preview": "Gemini-3-Pro-Preview",
        "minimax/minimax-m2": "MiniMax-M2",
        "mistral/magistral-medium": "Mistral-Magistral-Medium",
        "mistral/mistral-large": "Mistral-Large",
        "moonshotai/kimi-k2-thinking-turbo": "Kimi-K2-Thinking-Turbo",
        "openai/gpt-5-pro": "GPT-5-Pro",
        "openai/gpt-5.1": "GPT-5.1",
        "xai/grok-4": "Grok-4",
        "zai/glm-4.6": "GLM-4.6",
    }

    # Build unified dict: {qid: {renamed_model: 0/1}}
    unified = {}

    # Add supaihq data
    for qid, model_results in supaihq_data.items():
        if qid not in unified:
            unified[qid] = {}
        for model_name, score in model_results.items():
            renamed = MODEL_RENAMES.get(model_name, model_name)
            unified[qid][renamed] = score

    # Add deepwriter data (Gemini 3 Pro) for questions not in supaihq
    # or merge if the model is already present
    deepwriter_model = "Gemini-3-Pro-Preview"
    added_from_deepwriter = 0
    for qid, score in deepwriter_data.items():
        if qid not in unified:
            unified[qid] = {}
        if deepwriter_model not in unified[qid]:
            unified[qid][deepwriter_model] = score
            added_from_deepwriter += 1

    print(f"    Additional Gemini-3-Pro-Preview items from deepwriter: {added_from_deepwriter}")

    # Collect all unique question IDs and model names
    all_qids = sorted(unified.keys())
    all_models = set()
    for qid_results in unified.values():
        all_models.update(qid_results.keys())
    all_models = sorted(all_models)

    print(f"    Total unique questions: {len(all_qids)}")
    print(f"    Total unique models: {len(all_models)}")

    # -------------------------------------------------------------------------
    # 4. Build the full response matrix (models x items)
    # -------------------------------------------------------------------------
    print(f"\n[4] Building response matrix...")

    # Create DataFrame with models as rows and question IDs as columns
    matrix_data = {}
    for model in all_models:
        row = {}
        for qid in all_qids:
            val = unified.get(qid, {}).get(model, np.nan)
            row[qid] = val
        matrix_data[model] = row

    df_full = pd.DataFrame(matrix_data).T
    df_full.index.name = "Model"

    # Print full matrix stats
    total_cells = df_full.shape[0] * df_full.shape[1]
    filled_cells = df_full.notna().sum().sum()
    fill_rate = filled_cells / total_cells
    correct_cells = (df_full == 1).sum().sum()
    incorrect_cells = (df_full == 0).sum().sum()

    print(f"    Full matrix dimensions: {df_full.shape[0]} models x {df_full.shape[1]} items")
    print(f"    Total cells: {total_cells:,}")
    print(f"    Filled cells: {int(filled_cells):,} ({fill_rate*100:.1f}%)")
    print(f"    Correct (1): {int(correct_cells):,} ({correct_cells/filled_cells*100:.1f}% of filled)")
    print(f"    Incorrect (0): {int(incorrect_cells):,} ({incorrect_cells/filled_cells*100:.1f}% of filled)")

    # Per-model stats
    print(f"\n    Per-model statistics:")
    for model in all_models:
        row = df_full.loc[model]
        n_eval = row.notna().sum()
        n_correct = (row == 1).sum()
        accuracy = n_correct / n_eval * 100 if n_eval > 0 else 0
        print(f"      {model:45s}  evaluated={int(n_eval):5d}  correct={int(n_correct):4d}  accuracy={accuracy:.1f}%")

    # Save full matrix
    full_output = os.path.join(PROCESSED_DIR, "response_matrix.csv")
    df_full.to_csv(full_output)
    print(f"\n    Saved full matrix: {full_output}")

    # -------------------------------------------------------------------------
    # 5. Build dense matrix (filter to well-covered models and items)
    # -------------------------------------------------------------------------
    print(f"\n[5] Building dense response matrix (models with >= 50 items)...")

    # Filter models with at least 50 evaluated items
    MIN_ITEMS = 50
    model_eval_counts = df_full.notna().sum(axis=1)
    dense_models = model_eval_counts[model_eval_counts >= MIN_ITEMS].index.tolist()

    df_dense = df_full.loc[dense_models]

    # Also filter items: keep items that have at least 2 model evaluations
    item_eval_counts = df_dense.notna().sum(axis=0)
    dense_items = item_eval_counts[item_eval_counts >= 2].index.tolist()
    df_dense = df_dense[dense_items]

    dense_total = df_dense.shape[0] * df_dense.shape[1]
    dense_filled = df_dense.notna().sum().sum()
    dense_fill_rate = dense_filled / dense_total if dense_total > 0 else 0
    dense_correct = (df_dense == 1).sum().sum()

    print(f"    Dense matrix dimensions: {df_dense.shape[0]} models x {df_dense.shape[1]} items")
    print(f"    Total cells: {dense_total:,}")
    print(f"    Filled cells: {int(dense_filled):,} ({dense_fill_rate*100:.1f}%)")
    print(f"    Correct (1): {int(dense_correct):,} ({dense_correct/dense_filled*100:.1f}% of filled)")

    dense_output = os.path.join(PROCESSED_DIR, "response_matrix_dense.csv")
    df_dense.to_csv(dense_output)
    print(f"    Saved dense matrix: {dense_output}")

    # -------------------------------------------------------------------------
    # 6. Build model summary
    # -------------------------------------------------------------------------
    print(f"\n[6] Building model summary...")

    # Load SEAL leaderboard for reference
    seal_path = os.path.join(RAW_DIR, "seal_leaderboard_scores.csv")
    seal_df = pd.read_csv(seal_path)

    summary_rows = []
    for model in all_models:
        row_data = df_full.loc[model]
        n_eval = int(row_data.notna().sum())
        n_correct = int((row_data == 1).sum())
        n_incorrect = int((row_data == 0).sum())
        accuracy = n_correct / n_eval * 100 if n_eval > 0 else 0.0

        summary_rows.append({
            "model": model,
            "n_items_evaluated": n_eval,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy_pct": round(accuracy, 2),
            "source": "supaihq+deepwriter",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("accuracy_pct", ascending=False)

    summary_output = os.path.join(PROCESSED_DIR, "model_summary.csv")
    summary_df.to_csv(summary_output, index=False)
    print(f"    Saved model summary: {summary_output}")

    # -------------------------------------------------------------------------
    # 7. Final summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Data sources:")
    print(f"    1. supaihq/hle (GitHub): 19 models, 1,369 questions (per-item)")
    print(f"    2. deepwriter-ai/hle-gemini-3-0 (GitHub): 1 model, 878 questions (per-item)")
    print(f"    3. Scale AI SEAL leaderboard: 44 models, aggregate scores only")
    print(f"\n  Response matrix (full):")
    print(f"    Dimensions: {df_full.shape[0]} models x {df_full.shape[1]} items")
    print(f"    Fill rate: {fill_rate*100:.1f}%")
    print(f"    Value distribution:")
    print(f"      1 (correct):     {int(correct_cells):,} ({correct_cells/filled_cells*100:.1f}%)")
    print(f"      0 (incorrect):   {int(incorrect_cells):,} ({incorrect_cells/filled_cells*100:.1f}%)")
    print(f"      NaN (missing):   {int(total_cells - filled_cells):,} ({(1-fill_rate)*100:.1f}%)")
    print(f"\n  Response matrix (dense, models with >= {MIN_ITEMS} items):")
    print(f"    Dimensions: {df_dense.shape[0]} models x {df_dense.shape[1]} items")
    print(f"    Fill rate: {dense_fill_rate*100:.1f}%")
    print(f"\n  Note: HLE has 2,500 total questions. Per-item data covers")
    print(f"  {len(all_qids)} unique questions ({len(all_qids)/2500*100:.1f}% of full benchmark).")
    print(f"  The remaining questions have only aggregate scores available")
    print(f"  from the Scale AI SEAL leaderboard (saved in raw/ for reference).")

    print(f"\n  Output files:")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        fpath = os.path.join(PROCESSED_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f:45s}  {size_kb:.1f} KB")


if __name__ == "__main__":
    build_response_matrix()

    # Generate visualizations, then convert to .pt and upload to HuggingFace Hub
    # (set NO_UPLOAD=1 to skip the upload; .pt file is still generated)
    import subprocess
    _scripts = Path(__file__).resolve().parent.parent / "scripts"
    _bench = Path(__file__).resolve().parent.name
    subprocess.run([sys.executable, str(_scripts / "visualize_response_matrix.py"), _bench], check=False)
    _cmd = [sys.executable, str(_scripts / "upload_to_hf.py"), _bench]
    if os.environ.get("NO_UPLOAD") == "1":
        _cmd.append("--no-upload")
    subprocess.run(_cmd, check=False)
