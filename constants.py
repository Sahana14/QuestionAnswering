ques_words  = ["where", "when", "what", "how", "why", "who", "whose", "which","whom"]
ans_type = {"where":"location", "who":"person", "what":"organization","how jj":"numeric", "when":"date time numeric","why":"description",
            "how mod":"description", "whose":"person","which":"location", "how long": "numeric", "how far": "numeric", "how much": "numeric",
            "how old": "numeric","how many": "numeric", "how often": "numeric", "how big": "numeric", "how tall": "numeric", "how": "description"} #"what np":"entity"

MONTH = [ "january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

TIME = ["nowadays", "these days","lately","recently","last night","last week","last month",
        "last year","ago","since","previously","recently","just","in a week","in a moment",
        "in 3 days","next week","next month","next year","next summer","soon","presently",
        "in the end","eventually","at the end","finally","once","by the time","after","before",
        "for","on time","until","during","whenever","forever","by friday","by next week",
        "by the 7th of","immediately","January", "February", "March", "April", "May", "June",
        "July", "August","September", "October", "November", "December","Monday", "Tuesday",
        "Wednesday","Thursday","Friday","Saturday","Sunday","pm","p.m","am","a.m","o'clock","o clock"]

TIME1 = ["last night","last week","last month","last year","in a week","in 3 days","next week","next month","next year","next summer","on time","by friday","by next week",
        "by the 7th of","Monday", "Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","January", "February", "March", "April", "May", "June","July", "August","September", "October", "November", "December","pm","p.m","am","a.m","o'clock","o clock","years","year"]

LOCATION = ["Afghanistan","Albania","Algeria","American Samoa","Andorra","Angola","Anguilla","Antarctica","Antigua and Barbuda","Argentina",
"Armenia","Aruba","Australia","Austria","Azerbaijan","Bahamas","Bahrain","Bangladesh","Barbados","Belarus","Belgium","Belize","Benin",
"Bermuda","Bhutan","Bolivia","Bosnia and Herzegovina","Botswana","Brazil","Brunei Darussalam","Bulgaria","Burkina Faso","Burundi",
"Cambodia","Cameroon","Canada","Cape Verde","Cayman Islands","Central African Republic","Chad","Chile","China","Christmas Island",
"Cocos (Keeling) Islands","Colombia","Comoros","Democratic Republic of the Congo (Kinshasa)","Congo, Republic of(Brazzaville)",
"Cook Islands","Costa Rica","Ivory Coast","Croatia","Cuba","Cyprus","Czech Republic","Denmark","Djibouti","Dominica","Dominican Republic",
"East Timor (Timor-Leste)","Ecuador","Egypt","El Salvador","Equatorial Guinea","Eritrea","Estonia","Ethiopia","Falkland Islands","Faroe Islands",
"Fiji","Finland","France","French Guiana","French Polynesia","French Southern Territories","Gabon","Gambia","Georgia","Germany","Ghana","Gibraltar",
"Great Britain","Greece","Greenland","Grenada","Guadeloupe","Guam","Guatemala","Guinea","Guinea-Bissau","Guyana","","Haiti","Holy See","Honduras","Hong Kong",
"Hungary","Iceland","India","Indonesia","Iran (Islamic Republic of)","Iraq","Ireland","Israel","Italy","","Jamaica","Japan","Jordan","","","Kazakhstan",
"Kenya","Kiribati","Korea, Democratic People's Rep. (North Korea)","Korea, Republic of (South Korea)","Kosovo","","Kuwait","Kyrgyzstan","",
"Lao, People's Dem1ocratic Republic","Latvia","Lebanon","Lesotho","Liberia","Libya","Liechtenstein","Lithuania","Luxembourg","Macau","Macedonia, Rep. of",
"Madagascar","Malawi","Malaysia","Maldives","Mali","Malta","Marshall Islands","Martinique","Mauritania","Mauritius","Mayotte","Mexico",
"Micronesia, Federal States of","Moldova, Republic of","Monaco","Mongolia","Montenegro","Montserrat","Morocco","Mozambique","Myanmar, Burma","Namibia",
"Nauru","Nepal","Netherlands","Netherlands Antilles","New Caledonia","New Zealand","Nicaragua","Niger","Nigeria","Niue","Northern Mariana Islands","Norway",
"Oman","Pakistan","Palau","Palestinian territories","Panama","Papua New Guinea","Paraguay","Peru","Philippines","Pitcairn Island","Poland","Portugal","Puerto Rico",
"Qatar","Reunion Island","Romania","Russian Federation","Rwanda","Saint Kitts and Nevis","Saint Lucia","Saint Vincent and the Grenadines","Samoa","San Marino",
"Sao Tome and Principe","Saudi Arabia","Senegal","Serbia","Seychelles","Sierra Leone","Singapore","Slovakia (Slovak Republic)","Slovenia","Solomon Islands","Somalia",
"South Africa","South Sudan","Spain","Sri Lanka","Sudan","Suriname","Swaziland","Sweden","Switzerland","Syria, Syrian Arab Republic","Taiwan (Republic of China)",
"Tajikistan","Tanzania; officially the United Republic of Tanzania","Thailand","Tibet","Timor-Leste (East Timor)","Togo","Tokelau","Tonga","Trinidad and Tobago",
"Tunisia","Turkey","Turkmenistan","Turks and Caicos Islands","Tuvalu","Ugandax","Ukraine","United Arab Emirates","United Kingdom","United States","Uruguay","Uzbekistan",
"Vanuatu","Vatican City State (Holy See)","Venezuela","Vietnam","Virgin Islands (British)","Virgin Islands (U.S.)","Wallis and Futuna Islands","Western Sahara","Yemen",
"Zambia","Zimbabwe","Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","Florida","Georgia","Hawaii","Idaho","Illinois Indiana","Iowa",
"Kansas","Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana Nebraska","Nevada","New Hampshire","New Jersey",
"New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah",
"Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]

OCCUPATION = ["accountant", "actor", "actress", "actuary", "advisor", "aide", "ambassador", "animator", "archer", "athlete", "artist", "astronaut", "astronomer", "attorney", "auctioneer", "author", "babysitter", "baker", "ballerina", "banker", "barber", "baseball player", "basketball player", "bellhop", "blacksmith", "bookkeeper", "biologist", "bowler", "builder", "butcher", "butler", "cab driver", "calligrapher", "captain", "cardiologist", "carpenter", "cartographer", "cartoonist", "cashier", "catcher", "caterer", "cellist", "chaplain", "chef", "chemist", "chauffeur", "clerk", "coach", "cobbler", "composer", "concierge", "consul", "contractor", "cook", "cop", "coroner", "courier", "cryptographer", "custodian", "dancer", "dentist", "deputy", "dermatologist", "designer", "detective", "dictator", "director", "disc jockey", "diver", "doctor", "doorman", "driver", "drummer", "drycleaner", "ecologist", "economist", "editor", "educator", "electrician", "empress", "emperor", "engineer", "entomologist", "entrepeneur", "executive", "explorer", "exporter", "exterminator", "extra (in a movie)", "falconer", "farmer", "financier", "firefighter", "fisherman", "flutist", "football player", "foreman", "game designer", "garbage man", "gardener", "gatherer", "gemcutter", "geneticist", "general", "geologist", "geographer", "golfer", "governor", "grocer", "guide", "hairdresser", "handyman", "harpist", "highway patrol", "hobo", "hunter", "illustrator", "importer", "instructor", "intern", "internist", "inventor", "investigator", "jailer", "janitor", "jewler", "jester", "jockey", "journalist", "judge", "karate teacher", "laborer", "landlord", "landscaper", "laundress", "lawyer", "lecturer", "legal aide", "librarian", "librettist", "lifeguard", "linguist", "lobbyist", "locksmith", "lyricist", "magician", "maid", "mail carrier", "manager", "manufacturer", "marine", "marketer", "mason", "mathematician", "mayor", "mechanic", "messenger", "midwife", "miner", "model", "monk", "miralist", "musician", "navigator", "negotiator", "notary", "novelist", "nun", "nurse", "oboist", "operator", "optician", "oracle", "orderly", "ornithologist", "painter", "paleontologist", "pararlegal", "park ranger", "pathologist", "pawnbroker", "peddler", "pediatrician", "percussionist", "pharmacist", "philanthropist", "philosopher", "photographer", "physician", "physicist", "pianist", "pilot", "pitcher", "plumber", "poet", "police", "policeman", "policewoman", "politician", "president", "prince", "princess", "principal", "private", "private detective", "producer", "programmer", "professor", "psychiatrist psychologist", "publisher", "quarterback", "quilter", "radiologist", "rancher", "ranger", "real estate agent", "receptionist", "referee", "registrar", "reporter", "representative", "researcher", "restauranteur", "retailer", "retiree", "sailor", "salesperson", "samurai", "saxophonist", "scuba diver", "scientist", "scout", "seamstress", "security guard", "senator", "sheriff", "smith", "singer", "soldier", "spy", "star", "statistician", "stockbroker", "street sweeper", "student", "surgeon", "surveyor", "swimmer", "tailor", "tax collector", "taxidermist", "taxi driver", "teacher", "technician", "tennis player", "test pilot", "tiler", "toolmaker", "trader", "trainer", "translator", "trash collector", "travel agent", "treasurer", "truck driver", "tutor", "typist", "umpire", "undertaker", "usher", "valet", "veteran", "veterinarian", "vicar", "violinist", "waiter", "waitress", "warden", "watchmaker", "weaver", "welder", "woodcarver", "wrangler", "writer", "xylophonist", "yodeler", "zookeeper", "zoologist","protestors", "volunteer", "guard", "Mr.", "Mrs."]
PERSON_NAMES = ["Kolla","A.J","Michael", "Jessica", "Christopher", "Ashley", "Matthew", "Emily", "Joshua", "Samantha", "Jacob", "Sarah", "Nicholas", "Amanda", "Andrew", "Brittany", "Daniel", "Elizabeth", "Tyler", "Taylor", "Joseph", "Megan", "Brandon", "Hannah", "David", "Kayla", "James", "Lauren", "Ryan", "Stephanie", "John", "Rachel", "Zachary", "Jennifer", "Justin", "Nicole", "William", "Alexis", "Anthony", "Victoria", "Robert", "Amber", "Jonathan", "Alyssa", "Austin", "Courtney", "Alexander", "Danielle", "Kyle", "Rebecca", "Kevin", "Jasmine", "Thomas", "Brianna", "Cody", "Katherine", "Jordan", "Alexandra", "Eric", "Madison", "Benjamin", "Morgan", "Aaron", "Melissa", "Christian", "Michelle", "Samuel", "Kelsey", "Dylan", "Chelsea", "Steven", "Anna", "Brian", "Kimberly", "Jose", "Tiffany", "Timothy", "Olivia", "Nathan", "Mary", "Adam", "Christina", "Richard", "Allison", "Patrick", "Abigail", "Charles", "Sara", "Sean", "Shelby", "Jason", "Heather", "Cameron", "Haley", "Jeremy", "Maria", "Mark", "Kaitlyn", "Stephen", "Laura", "Jesse", "Erin", "Juan", "Andrea", "Alex", "Natalie", "Travis", "Jordan", "Jeffrey", "Brooke", "Ethan", "Julia", "Caleb", "Emma", "Luis", "Vanessa", "Jared", "Erica", "Logan", "Sydney", "Hunter", "Kelly", "Trevor", "Kristen", "Bryan", "Katelyn", "Evan", "Marissa", "Paul", "Amy", "Taylor", "Crystal", "Kenneth", "Paige", "Connor", "Cassandra", "Dustin", "Gabrielle", "Noah", "Katie", "Carlos", "Caitlin", "Devin", "Lindsey", "Gabriel", "Destiny", "Ian", "Kathryn", "Nathaniel", "Jacqueline", "Gregory", "Shannon", "Derek", "Jenna", "Corey", "Angela", "Scott", "Savannah", "Jesus", "Mariah", "Bradley", "Alexandria", "Dakota", "Sierra", "Antonio", "Alicia", "Marcus", "Briana", "Blake", "Miranda", "Garrett", "Jamie", "Edward", "Catherine", "Luke", "Brittney", "Shawn", "Breanna", "Peter", "Grace", "Seth", "Monica", "Mitchell", "Sabrina", "Adrian", "Madeline", "Victor", "Caroline", "Miguel", "Molly", "Shane", "Erika", "Chase", "Mackenzie", "Isaac", "Leah", "Spencer", "Diana", "Lucas", "Whitney", "Jack", "Cheyenne", "Tanner", "Bailey", "Angel", "Christine", "Vincent", "Meghan", "Isaiah", "Lindsay", "Dalton", "Cynthia", "Brett", "Angelica", "George", "Margaret", "Alejandro", "Kaitlin", "Elijah", "Alexa", "Cory", "Hailey", "Cole", "Veronica", "Joel", "Melanie", "Erik", "Bianca", "Jake", "Ariel", "Mason", "Autumn", "Jorge", "Kristin", "Dillon", "Bethany", "Raymond", "Lisa", "Colton", "Kristina", "Ricardo", "Holly", "Casey", "Leslie", "Francisco", "Casey", "Brendan", "Chloe", "Devon", "April", "Keith", "Julie", "Colin", "Claire", "Wesley", "Kaylee", "Phillip", "Brenda", "Oscar", "Kathleen", "Julian", "Rachael", "Johnathan", "Karen", "Eduardo", "Sophia", "Chad", "Patricia", "Donald", "Gabriela", "Bryce", "Kendra", "Ronald", "Dominique", "Alec", "Kara", "Dominic", "Desiree", "Grant", "Ana", "Martin", "Tara", "Henry", "Michaela", "Mario", "Brandi", "Xavier", "Carly", "Manuel", "Kylie", "Alan", "Karina", "Derrick", "Adriana", "Frank", "Valerie", "Tristan", "Caitlyn", "Collin", "Natasha", "Omar", "Hayley", "Jeremiah", "Rebekah", "Jackson", "Jocelyn", "Troy", "Cassidy", "Edgar", "Jade", "Javier", "Gabriella", "Douglas", "Makayla", "Clayton", "Daisy", "Jonathon", "Jillian", "Nicolas", "Alison", "Andre", "Audrey", "Maxwell", "Faith", "Philip", "Angel", "Ivan", "Nancy", "Levi", "Dana", "Sergio", "Krystal", "Roberto", "Alejandra", "Darius", "Ariana", "Andres", "Summer", "Cristian", "Mikayla", "Hector", "Isabella", "Fernando", "Raven", "Drew", "Katrina", "Curtis", "Kiara", "Gary", "Sandra", "Riley", "Meagan", "Johnny", "Kirsten", "Max", "Chelsey", "Dennis", "Lydia", "Malik", "Zoe", "Wyatt", "Monique", "Cesar", "Claudia", "Edwin", "Mallory", "Gavin", "Joanna", "Preston", "Deanna", "Marco", "Isabel", "Ruben", "Ashlee", "Allen", "Felicia", "Calvin", "Marisa", "Mathew", "Mckenzie", "Randy", "Mercedes", "Brent", "Jasmin", "Jerry", "Krista", "Hayden", "Yesenia", "Alexis", "Diamond", "Parker", "Evelyn", "Brady", "Cindy", "Tony", "Selena", "Pedro", "Brandy", "Craig", "Gina", "Larry", "Mia", "Erick", "Bridget", "Kaleb", "Tori", "Trenton", "Kassandra", "Carl", "Alisha", "Jeffery", "Anne", "Zachery", "Priscilla", "Raul", "Hope", "Sebastian", "Cierra", "Louis", "Maya", "Rafael", "Abby", "Marc", "Lacey", "Colby", "Denise", "Emmanuel", "Tabitha", "Jimmy", "Amelia", "Albert", "Jazmin", "Kristopher", "Allyson", "Julio", "Ashleigh", "Danny", "Cristina", "Harrison", "Hanna", "Russell", "Ciara", "Micah", "Colleen", "Armando", "Carolyn", "Jaime", "Karla", "Gerardo", "Meredith", "Diego", "Naomi", "Ricky", "Linda", "Chance", "Jazmine", "Gage", "Jaclyn", "Lawrence", "Asia", "Alberto", "Kendall", "Terry", "Daniela", "Chandler", "Britney", "Abraham", "Tessa", "Micheal", "Arianna", "Damian", "Teresa", "Conner", "Guadalupe", "Trey", "Tatiana", "Darren", "Adrianna", "Landon", "Nichole", "Skyler", "Nina", "Donovan", "Candace", "Andy", "Jada", "Rodney", "Kasey", "Lance", "Ellen", "Walter", "Katelynn", "Liam", "Lillian", "Joe", "Carmen", "Nolan", "Mayra", "Marcos", "Maggie", "Arthur", "Wendy", "Enrique", "Renee", "Todd", "Rosa", "Trent", "Raquel", "Bobby", "Aubrey", "Zackary", "Justine", "Jalen", "Tiara", "Dominique", "Alissa", "Billy", "Heidi", "Giovanni", "Susan", "Josiah", "Katlyn", "Josue", "Tamara", "Nickolas", "Theresa", "Jay", "Jordyn", "Roger", "Clarissa", "Ramon", "Ashlyn", "Ross", "Ruby", "Dallas", "Camille", "Deandre", "Kiana", "Jon", "Carrie", "Randall", "Esmeralda", "Jamal", "Kelli", "Arturo", "Riley", "Israel", "Angelina", "Alfredo", "Ebony", "Morgan", "Marina", "Theodore", "Jessie", "Maurice", "Jenny", "Malcolm", "Cecilia", "Jonah", "Alexus", "Dante", "Jacquelyn", "Bryant", "Miriam", "Carson", "Virginia", "Kody", "Candice", "Miles", "Cara", "Gerald", "Carolina", "Marvin", "Celeste", "Bailey", "Gloria", "Carter", "Marie", "Reginald", "Lily", "Willie", "Kelsie", "Branden", "Elise", "Eddie", "Tia", "Damon", "Charlotte", "Owen", "Janet", "Terrance", "Martha", "Avery", "Elena", "Jessie", "Cassie", "Ty", "Callie", "Lorenzo", "Kate", "Brock", "Shayla", "Jamie", "Alana", "Damien", "Brenna", "Johnathon", "Esther", "Tommy", "Dakota", "Lee", "Kylee", "Dominick", "Skylar", "Gustavo", "Savanna", "Frederick", "Alexia", "Elias", "Kyra", "Marquis", "Madeleine", "Aidan", "Vivian", "Shaun", "Stacy", "Salvador", "Carissa", "Roy", "Stacey", "Tevin", "Hillary", "Braden", "Ashton", "Demetrius", "Tiana", "Brennan", "Genesis", "Marshall", "Madelyn", "Conor", "Barbara", "Eli", "Ruth", "Jakob", "Alondra", "Zane", "Annie", "Quentin", "Sadie", "Dean", "Robin", "Ernesto", "Arielle", "Fabian", "Anastasia", "Angelo", "Janelle", "Pablo", "Brooklyn", "Ashton", "Rose", "Griffin", "Sharon", "Clinton", "Serena", "Brenden", "Logan", "Tyrone", "Tyler", "Terrell", "Pamela", "Bruce", "Helen", "Desmond", "Carla", "Kendall", "Kyla", "Quinton", "Toni", "Terrence", "Kennedy", "Steve", "Mckenna", "Zachariah", "Deja", "Darrell", "Sasha", "Zackery", "Liliana", "Wayne", "Kellie", "Simon", "Imani", "Drake", "Sophie", "Kelvin", "Eva", "Saul", "Natalia", "Leonardo", "Melinda", "Jerome", "Ann", "Rene", "Adrienne", "Nelson", "Aimee", "Franklin", "Sofia", "Myles", "Devon", "Francis", "Valeria", "Kameron", "Devin", "Ronnie", "Aaliyah", "Lane", "Juliana", "Byron", "Tanya", "Tucker", "Deborah", "Neil", "Marisol", "Jarrett", "Christian", "Orlando", "Melody", "Melvin", "Elisabeth", "Brendon", "Kierra", "Devonte", "Karissa", "Roman", "Nikki", "Ismael", "Kira", "Emilio", "Peyton", "Keegan", "Delaney", "Emanuel", "Sonia", "Quinn", "Chelsie", "Oliver", "Tina", "Darryl", "Tierra", "Forrest", "Payton", "Abel", "Nadia", "Kyler", "Shawna", "Esteban", "Isabelle", "Guillermo", "Makenzie", "Felix", "Josephine", "Cooper", "Sidney", "Leonard", "Mikaela", "Kendrick", "Macy", "Corbin", "Francesca", "Peyton", "Regina", "Dale", "Destinee", "Geoffrey", "Julianna", "Stanley", "Kali", "Karl", "Bryanna", "Weston", "Lorena", "Eugene", "Kailey", "Jermaine", "Brianne", "Ernest", "Avery", "Graham", "Tracy", "Wade", "Jane", "Keenan", "Bria", "Devante", "Cameron", "Warren", "Kayleigh", "Nikolas", "Alyson", "Bryson", "Taryn", "Glenn", "Emilee", "Moises", "Larissa", "Alvin", "Kristine", "Trevon", "Cortney", "Clay", "Alaina", "Skylar", "Alice", "Jordon", "Chasity", "Kurt", "Tayler", "Rodolfo", "Paola", "Cedric", "Gianna", "Beau", "Sylvia", "Dwayne", "Paula", "Hugo", "Kari", "Harley", "Precious", "Harold", "Leticia", "Jarrod", "Kaitlynn", "Devan", "Nicolette", "Tyrell", "Brittani", "Isiah", "Joy", "Denzel", "Shanice", "Dane", "Kristy", "Gilbert", "Robyn", "Sam", "Clara", "Dawson", "Harley", "Stefan", "Tania", "Ray", "India", "Harry", "Mariana", "Allan", "Giselle", "Joey", "Cheyanne", "Felipe", "Lucy", "Kenny", "Jeanette", "Lukas", "Frances", "Noel", "Josie", "Elliot", "Johanna", "Brayden", "Michele", "Alfonso", "Christy", "Kristian", "Noelle", "Jarred", "Alma", "Leon", "Shania", "Tyson", "Haylee", "Davis", "Allie", "Chris", "Kaila", "Ali", "Breana", "Braxton", "Carley", "Gilberto", "Irene", "Darian", "Daniella", "Rodrigo", "Jaime", "Brody", "Shayna", "Donte", "Simone", "Deshawn", "Marilyn", "Rudy", "Randi", "Javon", "Marlene", "Keaton", "Donna", "Elliott", "Angelique", "Khalil", "Abbey", "Jayson", "Ciera", "Charlie", "Elisa", "Sheldon", "Iris", "Reid", "Yvette", "Payton", "Kristi", "Rogelio", "Talia", "Davon", "Nora", "Grayson", "Christa", "Leo", "Skyler", "Tristen", "Kourtney", "Alfred", "Elaine", "Sterling", "Rachelle", "Rashad", "Ericka", "Ramiro", "Meaghan", "Marquise", "Hunter", "Ariel", "Misty", "Blaine", "Kiera", "Nathanael", "Stefanie", "Justice", "Katharine", "Tylor", "Trisha", "Cade", "Nia", "Marlon", "Kaleigh", "Jace", "Maritza", "Clarence", "Kenya", "Mackenzie", "Cristal", "Shaquille", "Tasha", "Lewis", "Carina", "Julius", "Corinne", "Tomas", "Maribel", "Darnell", "Blanca", "Garret", "Chantel", "Reed", "Julianne", "Ralph", "Ivy", "Tyree", "Tiffani", "Brad", "Judith", "Stuart", "Paris", "Stephan", "Kelsi", "Jaquan", "Alanna", "Roderick", "Haleigh", "Clifford", "Janae", "Daquan", "Tatyana", "Santiago", "Lauryn", "Alonzo", "Lizbeth", "Demarcus", "Yvonne", "Barry", "Charity", "Duncan", "Lori", "Quincy", "Carol", "Antoine", "Kelley", "Bernard", "Kiersten", "Noe", "Yasmin", "Wilson", "Diane", "Kelly", "Kaley", "Kurtis", "Eleanor", "Mauricio", "Aliyah", "Rolando", "Margarita", "Kory", "Julissa", "Bret", "Leanna", "Jarod", "Lyndsey", "Jaden", "Viviana", "Clint", "Jill", "Damion", "Sandy", "Kirk", "Chanel", "Lamar", "Eliza", "Howard", "Hallie", "Ahmad", "Shaina", "Earl", "Tianna", "Adan", "Mollie", "Dexter", "Sheila", "Derick", "Lesley", "Dorian", "Yolanda", "Shannon", "Emilie", "Maximilian", "Paulina", "Sidney", "Kaylin", "Deonte", "Desirae", "Nigel", "Casandra", "Efrain", "Moriah", "Colten", "Tanisha", "Jarvis", "Perla", "Ezekiel", "Kaylie", "Austen", "Anita", "Walker", "Genevieve", "Deon", "Tess", "Humberto", "Stephany", "Amir", "Maranda", "Jaylen", "Justice", "Darien", "Edith", "Mohammad", "Fatima", "Perry", "Shana", "Moses", "Alisa", "Carlton", "Lena", "Gordon", "Bonnie", "Courtney", "Beatriz", "Malachi", "Georgia", "Daryl", "Kassidy", "Deangelo", "Laurel", "Leroy", "Cayla", "Issac", "Micaela", "Kasey", "Lacy", "Markus", "Kathy", "Jaron", "Kacie", "Quintin", "Baylee", "Jamar", "Alina", "Ben", "Dawn", "Fredrick", "Antoinette", "Uriel", "Bridgette", "Dwight", "Roxanne", "Norman", "Yasmine", "Holden", "Alexandrea", "Sawyer", "Carlie", "Brice", "Chandler", "Chaz", "Rochelle", "Solomon", "Heaven", "Vicente", "Suzanne", "Alvaro", "Darian", "Korey", "Ava", "Deven", "Dorothy", "Osvaldo", "Araceli", "Mike", "Selina", "Shelby", "Sally", "Stephon", "Tyra", "Freddy", "Susana", "Addison", "Antonia", "Dion", "Alysha", "Brenton", "Katelin", "Heath", "Eileen", "Cullen", "Norma", "Mitchel", "Breanne", "Leonel", "Noemi", "Terence", "Iesha", "Salvatore", "Kaylyn", "Kareem", "Shauna", "Neal", "Rosemary", "Dashawn", "Keri", "Toby", "Celina", "Roland", "Ryan", "Glen", "Aisha", "Frankie", "Luz", "Joaquin", "Clare", "Coleman", "Lucia", "Aron", "Janice", "Mohammed", "Reagan", "Bennett", "Joyce", "Everett", "Mara", "Nathanial", "Hilary", "Irvin", "Charlene", "Shayne", "Rocio", "Jean", "Skye", "Baby", "Rhiannon", "Raheem", "Lea", "Clifton", "Nathalie", "Raphael", "Gillian", "Tre", "Jacklyn", "Jayden", "Ella", "Will", "Regan", "Triston", "Kiley", "Coty", "Aurora", "Darrin", "Shirley", "Kent", "Karli", "Lonnie", "Karly", "Cornelius", "Ashlie", "Reece", "Juanita", "Milton", "Keisha", "Conrad", "Angie", "Agustin", "Blair", "Kaden", "Mariela", "Mohamed", "Maegan", "Ezra", "Mckayla", "Darrius", "Celia", "Vernon", "Annette", "Marcel", "Loren", "Caden", "Trinity", "Reynaldo", "Yadira", "Fred", "Thalia", "Tate", "Reyna", "Dandre", "Katy", "Adolfo", "Eden", "Lamont", "Alex", "Tracy", "Ashlynn", "Jamel", "Madalyn", "Gunnar", "Shelbi", "Infant", "Leanne", "Rory", "Jodi", "Lloyd", "Hailee", "Clark", "Katarina", "Arnold", "Abbie", "Rigoberto", "Kerri", "Ignacio", "Essence", "Ahmed", "Tonya", "Rickey", "Sonya", "Cortez", "Jaqueline", "Darin", "Mya", "Don", "Darlene", "Kai", "Beverly", "Darion", "Keely", "Guadalupe", "Kasandra", "Duane", "Cora", "Kristofer", "Lizette", "Camron", "Katlin", "Travon", "Makenna", "Nestor", "Jena", "Juwan", "Shantel", "Deion", "Karlee", "Rick", "Phoebe", "Dillan", "Lexus", "Ezequiel", "Maura", "Sage", "Aileen", "Jonas", "Halie", "Jasper", "Dulce", "Kolby", "Sarai", "Anton", "Jesse", "Winston", "Elaina", "Jamison", "Rita", "Kerry", "Ashly", "German", "Ellie", "Antwan", "Micah", "Jovan", "Shanna", "Brennen", "Athena", "Tristin", "Arlene", "Dimitri", "Kacey", "Camden", "Kelsea", "Trace", "Rylee", "Herbert", "Ayanna", "Ulises", "Connie", "Pierce", "Justina", "Guy", "Silvia", "Gerard", "Tammy", "Brooks", "Alayna", "Thaddeus", "Kirstin", "Pierre", "Latoya", "Johnnie", "Sade", "Jefferson", "Rebeca", "Jameson", "Gretchen", "Isaias", "Beth", "Reese", "Melina", "Kobe", "Janessa", "Cordell", "Anissa", "Robin", "Camryn", "Jimmie", "Jenifer", "Houston", "Karlie", "Freddie", "Bailee", "Reuben", "Lara", "Bradford", "Kerry", "Cyrus", "Dallas", "Kade", "Brook", "Jairo", "Octavia", "Paris", "Joanne", "Elvis", "Shaniqua", "Tobias", "Elyse", "Moshe", "Addison", "Sammy", "Cheryl", "Estevan", "Gwendolyn", "Brandan", "Alycia", "Davion", "Kailee", "Colt", "Jayla", "Kolton", "Jana", "Davonte", "Khadijah", "Donnie", "Debra", "Ladarius", "Fiona", "Trever", "Savanah", "Rhett", "Ali", "Santos", "Kori", "Kane", "Yessenia", "August", "Brittni", "Madison", "Corina", "Octavio", "Constance", "Bo", "Lexi", "Heriberto", "Mandy", "Cruz", "Carli", "Rashawn", "Christie", "Donavan", "Elissa", "Zechariah", "Christiana", "Keon", "Shelbie", "Nico", "Carlee", "Gino", "Allyssa", "Dayton", "Kaela", "Akeem", "Tabatha", "Jerrod", "Fabiola", "Hassan", "Dianna", "Jackie", "Kalyn", "Dangelo", "Mindy", "Hakeem", "Shakira", "Brandyn", "Shyanne", "Aldo", "Nikita", "Jaylon", "Alysia", "Elmer", "Kaci", "Aubrey", "Latasha", "Garrison", "Kristie", "Cristopher", "Shea", "Tyrese", "Maddison", "Dewayne", "Gladys", "Bradly", "Jalisa", "Bernardo", "Stacie", "Kellen", "Montana", "Nick", "Hollie", "Kieran", "Nadine", "Dequan", "Maricela", "Herman", "Graciela", "Coby", "Helena", "Dylon", "Stevie", "Myron", "Halle", "Dillion", "Maureen", "Gene", "Janette", "Marquez", "Brielle", "Elisha", "Destini", "Tariq", "Janie", "Junior", "Kala", "Arron", "Demi", "Silas", "Baby", "Dana", "Ingrid", "Loren", "Audra", "Lester", "Jayme", "Zakary", "Chantal", "Alonso", "Breann", "Gonzalo", "Destiney", "River", "Alessandra", "Raymundo", "Daphne", "Rico", "Bobbie", "Aric", "Tamia", "Sonny", "Jessika", "Devyn", "Leigh", "Royce", "Adrian", "Benny", "Liana", "Josh", "Kristian", "Talon", "Latisha", "Benito", "Devan", "Vance", "Macey", "Irving", "Myra", "Ellis", "Judy", "Marques", "Nataly", "Hugh", "Sage", "Asa", "Jennie", "Josef", "Cecelia", "Grady", "Dayna", "Misael", "Gabriel", "Shea", "Lucero", "Leland", "Aspen", "Jaleel", "Magdalena", "Jeff", "Joelle", "Jabari", "Ayla", "Jamarcus", "Devyn", "Ulysses", "Jami", "Jamil", "Kallie", "Bronson", "Princess", "Leslie", "Berenice", "Fidel", "Brandie", "Quinten", "Cori", "Efren", "Christen", "Stewart", "Leann", "Vaughn", "Alysa", "Dontae", "Racheal", "Mikel", "Ayana", "Devonta", "Staci", "Dan", "Elisha", "Lincoln", "Kaycee", "Mateo", "Kristyn", "Tyron", "Kortney", "Jacoby", "Jackie", "Demarco", "Terri", "Julien", "Tracey", "Ari", "Darby", "Floyd", "Itzel", "Antony", "Bobbi", "Anderson", "Kayley", "Emmett", "Martina", "Alexandro", "Krysta", "Dusty", "Shelly", "Rocky", "Catalina", "Dario", "Lacie", "Barrett", "Kalie", "Remington", "Emilia", "Rex", "Chaya", "Kevon", "Karley", "Titus", "Elsa", "Keven", "Bryana", "Austyn", "Noel", "Hernan", "Domonique", "Codey", "Iliana", "Jaylin", "Asha", "Clyde", "Tatum", "Orion", "Juana", "Alton", "Monika", "Davin", "Celine", "Donnell", "Joselyn", "Jorden", "Myranda", "Axel", "Dalia", "Hans", "Annika", "Ibrahim", "Damaris", "Kadeem", "Marcella", "Chaim", "Kianna", "Asher", "Lesly", "Garett", "Jaimie", "Jade", "Marisela", "Muhammad", "Melisa", "Broderick", "Giovanna", "Carlo", "Britany", "Gregorio", "Betty", "Andreas", "Emerald", "Raekwon", "Kailyn", "Deshaun", "Lexie", "Lionel", "Carson", "Maximillian", "Eliana", "Cecil", "Kirstie", "Erich", "Patrice", "Aiden", "Kaylynn", "Javonte", "Infant", "Rasheed", "Sydnee", "Dejuan", "Beatrice", "Edgardo", "Zoey", "Adonis", "Brittanie", "Niko", "Breonna", "Denver", "Marlena", "Jim", "Hali", "Jamaal", "Lynn", "Dakotah", "Malia", "Abram", "Roxana", "Justyn", "Kia", "Keanu", "Alecia", "Jacques", "Scarlett", "Justus", "Belinda", "Brant", "Lourdes", "Chadwick", "Jazmyn", "Augustus", "Mariam", "Montana", "Terra", "Zion", "Pauline", "Jessy", "Emely", "Harvey", "Christin", "Blair", "Ivette", "Cale", "Eboni", "Tyshawn", "Lyric", "Bryon", "Rikki", "Dallin", "Delia", "Amos", "Brooklynn", "Deondre", "Dina", "Valentin", "Macie", "Darrion", "Aubree", "Kenton", "Sherry", "Erin", "Leila", "Kole", "Alesha", "Layne", "Brionna", "Eliseo", "Billie", "Randolph", "Griselda", "Codie", "Ashanti", "Armani", "Kayli", "Giancarlo", "Tricia", "Stone", "Hana", "Ted", "Aleah", "Tristian", "Aja", "Ryne", "Elyssa", "Gunner", "Yasmeen", "Alexandre", "Traci", "Najee", "Miracle", "Domenic", "Mikala", "Ervin", "Alena", "Unknown", "Anjelica", "Kelsey", "Danica", "Tory", "Isis", "Kelton", "Susanna", "Tyquan", "Estefania", "Eddy", "Drew", "Jess", "Jocelyne", "Destin", "Sienna", "Kirby", "Lilly", "Khalid", "Janay", "Cristobal", "Katlynn", "Francesco", "Lorraine", "Hudson", "Lizeth", "Wendell", "Mireya", "Isidro", "Kimberlee", "Jody", "Abbigail", "Darrian", "Kendal", "Braeden", "Layla", "Rusty", "Shyann", "Galen", "Blake", "Lyle", "Ashely", "Chester", "Jailene", "Keshawn", "Leandra", "Abdul", "Corey", "Tavon", "Katerina", "Easton", "Kaelyn", "Justen", "Kenia", "Marcelo", "Yazmin", "Dionte", "Sheena", "Turner", "Kenzie", "Maverick", "Sarina", "Storm", "Stormy", "Daron", "Kamryn", "Shelton", "Leilani", "Mickey", "Anika", "Syed", "Lana", "Edmund", "Esperanza", "Spenser", "Maira", "Devontae", "Annabelle", "Emerson", "Ashli", "Kennedy", "Chelsi", "Auston", "Gracie", "Samir", "Jessi", "Greg", "Kinsey", "Tom", "Eryn", "Forest", "Marley", "Markel", "Irma", "Braydon", "Leeann", "Rohan", "Reina", "Trae", "Jerrica", "Romeo", "Haylie", "Dominque", "Kimberley", "Tayler", "Tiera", "Demario", "Tiffanie", "Kegan", "Chyna", "Kelby", "Zaria", "Jerrell", "Maia", "Ryder", "Chantelle", "Greyson", "Juliette", "Keyshawn", "Kristal", "Alden", "Amari", "Brayan", "Amie", "Antwon", "Dasia", "Shamar", "Unique", "Draven", "Olga", "Koby", "Nayeli", "Jerod", "Juliet", "Pete", "Rhonda", "Rodrick", "Alivia", "Ashley", "Marlee", "Darrien", "Kyleigh", "Deontae", "Tyesha", "Otis", "Sydni", "Shaquan", "Patience", "Randal", "Krystle", "Nasir", "Tatianna", "Scotty", "Janine", "Wilfredo", "Valentina", "Gianni", "Daisha", "Sherman", "Kathrine", "Darwin", "Laurie", "Seamus", "Sydnie", "Genaro", "Jamila", "Samson", "Mattie", "Morris", "Jodie", "Jaxon", "Jasmyn", "Markell", "Tamera", "Ronaldo", "Madyson", "Waylon", "Anahi", "Brennon", "Laken", "Derik", "Lakeisha"]
locPrep = ["in", "outside", "on", "between", "at", "beside", "by", "beyond", "near", "in front of", "nearby", "in back of", "above", "behind", "below", "next to", "over", "on top of", "under", "within", "up", "beneath", "down", "underneath", "around", "among", "through", "along", "inside", "against"]

#score constants
clue = 3
good_clue = 4
confident = 6
slam_dunk = 20