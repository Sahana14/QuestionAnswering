import sys
import os, glob
import string
import nltk
import re
import textwrap
import copy
import numpy
nltk.data.path.append("/home/sandeep/nltk_data")
from nltk.corpus import state_union
java_path = "C:/Program Files/Java/jdk1.8.0_60/bin/java.exe"
os.environ['JAVAHOME'] = java_path
nltk.internals.config_java("C:/Program Files/Java/jdk1.8.0_60/bin/java.exe")
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordNERTagger
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
lemmatizer = nltk.WordNetLemmatizer()
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]
quoted = re.compile('"([^"]*)"')
#with open(sys.argv[1], 'r') as f:
   # contents = f.readlines()
ques_words  = ["where", "when", "what", "how", "why", "who", "whose", "which"]
ans_type = {"where":"location", "who":"person", "what":"organization", "how jj":"numeric", "when":"date time numeric",
            "why":"description", "how mod":"description", "whose":"person",
            "which":"location", "how long": "numeric", "how far": "numeric", "how much": "numeric",
            "how old": "numeric", "how often": "numeric"} #"what np":"entity"

MONTH = [ "january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december"]

TIME = ["nowadays", "these days","lately","recently","last night","last week","last month",
        "last year","ago","since","previously","recently","just","in a week","in a moment",
        "in 3 days","next week","next month","next year","next summer","soon","presently",
        "in the end","eventually","at the end","finally","once","by the time","after","before",
        "for","on time","until","during","whenever","forever","by friday","by next week",
        "by the 7th of","immediately","january", "february", "march", "april", "may", "june",
        "july", "august","september", "october", "november", "december","pm","p.m","am","a.m","o'clock","o clock"]

TIME1 = ["last night","last week","last month","last year","in a week","in 3 days","next week","next month","next year","next summer","once","by the time","on time","until","during","whenever","forever","by friday","by next week",
        "by the 7th of","january", "february", "march", "april", "may", "june","july", "august","september", "october", "november", "december","pm","p.m","am","a.m","o'clock","o clock"]

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

#OCCUPATION = ["accountant", "actor", "actress", "actuary", "advisor", "aide", "ambassador", "animator", "archer", "athlete", "artist", "astronaut", "astronomer", "attorney", "auctioneer", "author", "babysitter", "baker", "ballerina", "banker", "barber", "baseball player", "basketball player", "bellhop", "blacksmith", "bookkeeper", "biologist", "bowler", "builder", "butcher", "butler", "cab driver", "calligrapher", "captain", "cardiologist", "carpenter", "cartographer", "cartoonist", "cashier", "catcher", "caterer", "cellist", "chaplain", "chef", "chemist", "chauffeur", "clerk", "coach", "cobbler", "composer", "concierge", "consul", "contractor", "cook", "cop", "coroner", "courier", "cryptographer", "custodian", "dancer", "dentist", "deputy", "dermatologist", "designer", "detective", "dictator", "director", "disc jockey", "diver", "doctor", "doorman", "driver", "drummer", "drycleaner", "ecologist", "economist", "editor", "educator", "electrician", "empress", "emperor", "engineer", "entomologist", "entrepeneur", "executive", "explorer", "exporter", "exterminator", "extra (in a movie)", "falconer", "farmer", "financier", "firefighter", "fisherman", "flutist", "football player", "foreman", "game designer", "garbage man", "gardener", "gatherer", "gemcutter", "geneticist", "general", "geologist", "geographer", "golfer", "governor", "grocer", "guide", "hairdresser", "handyman", "harpist", "highway patrol", "hobo", "hunter", "illustrator", "importer", "instructor", "intern", "internist", "inventor", "investigator", "jailer", "janitor", "jewler", "jester", "jockey", "journalist", "judge", "karate teacher", "laborer", "landlord", "landscaper", "laundress", "lawyer", "lecturer", "legal aide", "librarian", "librettist", "lifeguard", "linguist", "lobbyist", "locksmith", "lyricist", "magician", "maid", "mail carrier", "manager", "manufacturer", "marine", "marketer", "mason", "mathematician", "mayor", "mechanic", "messenger", "midwife", "miner", "model", "monk", "miralist", "musician", "navigator", "negotiator", "notary", "novelist", "nun", "nurse", "oboist", "operator", "optician", "oracle", "orderly", "ornithologist", "painter", "paleontologist", "pararlegal", "park ranger", "pathologist", "pawnbroker", "peddler", "pediatrician", "percussionist", "pharmacist", "philanthropist", "philosopher", "photographer", "physician", "physicist", "pianist", "pilot", "pitcher", "plumber", "poet", "police", "policeman", "policewoman", "politician", "president", "prince", "princess", "principal", "private", "private detective", "producer", "programmer", "professor", "psychiatrist psychologist", "publisher", "quarterback", "quilter", "radiologist", "rancher", "ranger", "real estate agent", "receptionist", "referee", "registrar", "reporter", "representative", "researcher", "restauranteur", "retailer", "retiree", "sailor", "salesperson", "samurai", "saxophonist", "scuba diver", "scientist", "scout", "seamstress", "security guard", "senator", "sheriff", "smith", "singer", "soldier", "spy", "star", "statistician", "stockbroker", "street sweeper", "student", "surgeon", "surveyor", "swimmer", "tailor", "tax collector", "taxidermist", "taxi driver", "teacher", "technician", "tennis player", "test pilot", "tiler", "toolmaker", "trader", "trainer", "translator", "trash collector", "travel agent", "treasurer", "truck driver", "tutor", "typist", "umpire", "undertaker", "usher", "valet", "veteran", "veterinarian", "vicar", "violinist", "waiter", "waitress", "warden", "watchmaker", "weaver", "welder", "woodcarver", "wrangler", "writer", "xylophonist", "yodeler", "zookeeper", "zoologist"]
OCCUPATION = ["accountant", "actor", "actress", "actuary", "advisor", "aide", "ambassador", "animator", "archer", "athlete", "artist", "astronaut", "astronomer", "attorney", "auctioneer", "author", "babysitter", "baker", "ballerina", "banker", "barber", "baseball player", "basketball player", "bellhop", "blacksmith", "bookkeeper", "biologist", "bowler", "builder", "butcher", "butler", "cab driver", "calligrapher", "captain", "cardiologist", "carpenter", "cartographer", "cartoonist", "cashier", "catcher", "caterer", "cellist", "chaplain", "chef", "chemist", "chauffeur", "clerk", "coach", "cobbler", "composer", "concierge", "consul", "contractor", "cook", "cop", "coroner", "courier", "cryptographer", "custodian", "dancer", "dentist", "deputy", "dermatologist", "designer", "detective", "dictator", "director", "disc jockey", "diver", "doctor", "doorman", "driver", "drummer", "drycleaner", "ecologist", "economist", "editor", "educator", "electrician", "empress", "emperor", "engineer", "entomologist", "entrepeneur", "executive", "explorer", "exporter", "exterminator", "extra (in a movie)", "falconer", "farmer", "financier", "firefighter", "fisherman", "flutist", "football player", "foreman", "game designer", "garbage man", "gardener", "gatherer", "gemcutter", "geneticist", "general", "geologist", "geographer", "golfer", "governor", "grocer", "guide", "hairdresser", "handyman", "harpist", "highway patrol", "hobo", "hunter", "illustrator", "importer", "instructor", "intern", "internist", "inventor", "investigator", "jailer", "janitor", "jewler", "jester", "jockey", "journalist", "judge", "karate teacher", "laborer", "landlord", "landscaper", "laundress", "lawyer", "lecturer", "legal aide", "librarian", "librettist", "lifeguard", "linguist", "lobbyist", "locksmith", "lyricist", "magician", "maid", "mail carrier", "manager", "manufacturer", "marine", "marketer", "mason", "mathematician", "mayor", "mechanic", "messenger", "midwife", "miner", "model", "monk", "miralist", "musician", "navigator", "negotiator", "notary", "novelist", "nun", "nurse", "oboist", "operator", "optician", "oracle", "orderly", "ornithologist", "painter", "paleontologist", "pararlegal", "park ranger", "pathologist", "pawnbroker", "peddler", "pediatrician", "percussionist", "pharmacist", "philanthropist", "philosopher", "photographer", "physician", "physicist", "pianist", "pilot", "pitcher", "plumber", "poet", "police", "policeman", "policewoman", "politician", "president", "prince", "princess", "principal", "private", "private detective", "producer", "programmer", "professor", "psychiatrist psychologist", "publisher", "quarterback", "quilter", "radiologist", "rancher", "ranger", "real estate agent", "receptionist", "referee", "registrar", "reporter", "representative", "researcher", "restauranteur", "retailer", "retiree", "sailor", "salesperson", "samurai", "saxophonist", "scuba diver", "scientist", "scout", "seamstress", "security guard", "senator", "sheriff", "smith", "singer", "soldier", "spy", "star", "statistician", "stockbroker", "street sweeper", "student", "surgeon", "surveyor", "swimmer", "tailor", "tax collector", "taxidermist", "taxi driver", "teacher", "technician", "tennis player", "test pilot", "tiler", "toolmaker", "trader", "trainer", "translator", "trash collector", "travel agent", "treasurer", "truck driver", "tutor", "typist", "umpire", "undertaker", "usher", "valet", "veteran", "veterinarian", "vicar", "violinist", "waiter", "waitress", "warden", "watchmaker", "weaver", "welder", "woodcarver", "wrangler", "writer", "xylophonist", "yodeler", "zookeeper", "zoologist","protestors"]
PERSON_NAMES = ["Michael", "Jessica", "Christopher", "Ashley", "Matthew", "Emily", "Joshua", "Samantha", "Jacob", "Sarah", "Nicholas", "Amanda", "Andrew", "Brittany", "Daniel", "Elizabeth", "Tyler", "Taylor", "Joseph", "Megan", "Brandon", "Hannah", "David", "Kayla", "James", "Lauren", "Ryan", "Stephanie", "John", "Rachel", "Zachary", "Jennifer", "Justin", "Nicole", "William", "Alexis", "Anthony", "Victoria", "Robert", "Amber", "Jonathan", "Alyssa", "Austin", "Courtney", "Alexander", "Danielle", "Kyle", "Rebecca", "Kevin", "Jasmine", "Thomas", "Brianna", "Cody", "Katherine", "Jordan", "Alexandra", "Eric", "Madison", "Benjamin", "Morgan", "Aaron", "Melissa", "Christian", "Michelle", "Samuel", "Kelsey", "Dylan", "Chelsea", "Steven", "Anna", "Brian", "Kimberly", "Jose", "Tiffany", "Timothy", "Olivia", "Nathan", "Mary", "Adam", "Christina", "Richard", "Allison", "Patrick", "Abigail", "Charles", "Sara", "Sean", "Shelby", "Jason", "Heather", "Cameron", "Haley", "Jeremy", "Maria", "Mark", "Kaitlyn", "Stephen", "Laura", "Jesse", "Erin", "Juan", "Andrea", "Alex", "Natalie", "Travis", "Jordan", "Jeffrey", "Brooke", "Ethan", "Julia", "Caleb", "Emma", "Luis", "Vanessa", "Jared", "Erica", "Logan", "Sydney", "Hunter", "Kelly", "Trevor", "Kristen", "Bryan", "Katelyn", "Evan", "Marissa", "Paul", "Amy", "Taylor", "Crystal", "Kenneth", "Paige", "Connor", "Cassandra", "Dustin", "Gabrielle", "Noah", "Katie", "Carlos", "Caitlin", "Devin", "Lindsey", "Gabriel", "Destiny", "Ian", "Kathryn", "Nathaniel", "Jacqueline", "Gregory", "Shannon", "Derek", "Jenna", "Corey", "Angela", "Scott", "Savannah", "Jesus", "Mariah", "Bradley", "Alexandria", "Dakota", "Sierra", "Antonio", "Alicia", "Marcus", "Briana", "Blake", "Miranda", "Garrett", "Jamie", "Edward", "Catherine", "Luke", "Brittney", "Shawn", "Breanna", "Peter", "Grace", "Seth", "Monica", "Mitchell", "Sabrina", "Adrian", "Madeline", "Victor", "Caroline", "Miguel", "Molly", "Shane", "Erika", "Chase", "Mackenzie", "Isaac", "Leah", "Spencer", "Diana", "Lucas", "Whitney", "Jack", "Cheyenne", "Tanner", "Bailey", "Angel", "Christine", "Vincent", "Meghan", "Isaiah", "Lindsay", "Dalton", "Cynthia", "Brett", "Angelica", "George", "Margaret", "Alejandro", "Kaitlin", "Elijah", "Alexa", "Cory", "Hailey", "Cole", "Veronica", "Joel", "Melanie", "Erik", "Bianca", "Jake", "Ariel", "Mason", "Autumn", "Jorge", "Kristin", "Dillon", "Bethany", "Raymond", "Lisa", "Colton", "Kristina", "Ricardo", "Holly", "Casey", "Leslie", "Francisco", "Casey", "Brendan", "Chloe", "Devon", "April", "Keith", "Julie", "Colin", "Claire", "Wesley", "Kaylee", "Phillip", "Brenda", "Oscar", "Kathleen", "Julian", "Rachael", "Johnathan", "Karen", "Eduardo", "Sophia", "Chad", "Patricia", "Donald", "Gabriela", "Bryce", "Kendra", "Ronald", "Dominique", "Alec", "Kara", "Dominic", "Desiree", "Grant", "Ana", "Martin", "Tara", "Henry", "Michaela", "Mario", "Brandi", "Xavier", "Carly", "Manuel", "Kylie", "Alan", "Karina", "Derrick", "Adriana", "Frank", "Valerie", "Tristan", "Caitlyn", "Collin", "Natasha", "Omar", "Hayley", "Jeremiah", "Rebekah", "Jackson", "Jocelyn", "Troy", "Cassidy", "Edgar", "Jade", "Javier", "Gabriella", "Douglas", "Makayla", "Clayton", "Daisy", "Jonathon", "Jillian", "Nicolas", "Alison", "Andre", "Audrey", "Maxwell", "Faith", "Philip", "Angel", "Ivan", "Nancy", "Levi", "Dana", "Sergio", "Krystal", "Roberto", "Alejandra", "Darius", "Ariana", "Andres", "Summer", "Cristian", "Mikayla", "Hector", "Isabella", "Fernando", "Raven", "Drew", "Katrina", "Curtis", "Kiara", "Gary", "Sandra", "Riley", "Meagan", "Johnny", "Kirsten", "Max", "Chelsey", "Dennis", "Lydia", "Malik", "Zoe", "Wyatt", "Monique", "Cesar", "Claudia", "Edwin", "Mallory", "Gavin", "Joanna", "Preston", "Deanna", "Marco", "Isabel", "Ruben", "Ashlee", "Allen", "Felicia", "Calvin", "Marisa", "Mathew", "Mckenzie", "Randy", "Mercedes", "Brent", "Jasmin", "Jerry", "Krista", "Hayden", "Yesenia", "Alexis", "Diamond", "Parker", "Evelyn", "Brady", "Cindy", "Tony", "Selena", "Pedro", "Brandy", "Craig", "Gina", "Larry", "Mia", "Erick", "Bridget", "Kaleb", "Tori", "Trenton", "Kassandra", "Carl", "Alisha", "Jeffery", "Anne", "Zachery", "Priscilla", "Raul", "Hope", "Sebastian", "Cierra", "Louis", "Maya", "Rafael", "Abby", "Marc", "Lacey", "Colby", "Denise", "Emmanuel", "Tabitha", "Jimmy", "Amelia", "Albert", "Jazmin", "Kristopher", "Allyson", "Julio", "Ashleigh", "Danny", "Cristina", "Harrison", "Hanna", "Russell", "Ciara", "Micah", "Colleen", "Armando", "Carolyn", "Jaime", "Karla", "Gerardo", "Meredith", "Diego", "Naomi", "Ricky", "Linda", "Chance", "Jazmine", "Gage", "Jaclyn", "Lawrence", "Asia", "Alberto", "Kendall", "Terry", "Daniela", "Chandler", "Britney", "Abraham", "Tessa", "Micheal", "Arianna", "Damian", "Teresa", "Conner", "Guadalupe", "Trey", "Tatiana", "Darren", "Adrianna", "Landon", "Nichole", "Skyler", "Nina", "Donovan", "Candace", "Andy", "Jada", "Rodney", "Kasey", "Lance", "Ellen", "Walter", "Katelynn", "Liam", "Lillian", "Joe", "Carmen", "Nolan", "Mayra", "Marcos", "Maggie", "Arthur", "Wendy", "Enrique", "Renee", "Todd", "Rosa", "Trent", "Raquel", "Bobby", "Aubrey", "Zackary", "Justine", "Jalen", "Tiara", "Dominique", "Alissa", "Billy", "Heidi", "Giovanni", "Susan", "Josiah", "Katlyn", "Josue", "Tamara", "Nickolas", "Theresa", "Jay", "Jordyn", "Roger", "Clarissa", "Ramon", "Ashlyn", "Ross", "Ruby", "Dallas", "Camille", "Deandre", "Kiana", "Jon", "Carrie", "Randall", "Esmeralda", "Jamal", "Kelli", "Arturo", "Riley", "Israel", "Angelina", "Alfredo", "Ebony", "Morgan", "Marina", "Theodore", "Jessie", "Maurice", "Jenny", "Malcolm", "Cecilia", "Jonah", "Alexus", "Dante", "Jacquelyn", "Bryant", "Miriam", "Carson", "Virginia", "Kody", "Candice", "Miles", "Cara", "Gerald", "Carolina", "Marvin", "Celeste", "Bailey", "Gloria", "Carter", "Marie", "Reginald", "Lily", "Willie", "Kelsie", "Branden", "Elise", "Eddie", "Tia", "Damon", "Charlotte", "Owen", "Janet", "Terrance", "Martha", "Avery", "Elena", "Jessie", "Cassie", "Ty", "Callie", "Lorenzo", "Kate", "Brock", "Shayla", "Jamie", "Alana", "Damien", "Brenna", "Johnathon", "Esther", "Tommy", "Dakota", "Lee", "Kylee", "Dominick", "Skylar", "Gustavo", "Savanna", "Frederick", "Alexia", "Elias", "Kyra", "Marquis", "Madeleine", "Aidan", "Vivian", "Shaun", "Stacy", "Salvador", "Carissa", "Roy", "Stacey", "Tevin", "Hillary", "Braden", "Ashton", "Demetrius", "Tiana", "Brennan", "Genesis", "Marshall", "Madelyn", "Conor", "Barbara", "Eli", "Ruth", "Jakob", "Alondra", "Zane", "Annie", "Quentin", "Sadie", "Dean", "Robin", "Ernesto", "Arielle", "Fabian", "Anastasia", "Angelo", "Janelle", "Pablo", "Brooklyn", "Ashton", "Rose", "Griffin", "Sharon", "Clinton", "Serena", "Brenden", "Logan", "Tyrone", "Tyler", "Terrell", "Pamela", "Bruce", "Helen", "Desmond", "Carla", "Kendall", "Kyla", "Quinton", "Toni", "Terrence", "Kennedy", "Steve", "Mckenna", "Zachariah", "Deja", "Darrell", "Sasha", "Zackery", "Liliana", "Wayne", "Kellie", "Simon", "Imani", "Drake", "Sophie", "Kelvin", "Eva", "Saul", "Natalia", "Leonardo", "Melinda", "Jerome", "Ann", "Rene", "Adrienne", "Nelson", "Aimee", "Franklin", "Sofia", "Myles", "Devon", "Francis", "Valeria", "Kameron", "Devin", "Ronnie", "Aaliyah", "Lane", "Juliana", "Byron", "Tanya", "Tucker", "Deborah", "Neil", "Marisol", "Jarrett", "Christian", "Orlando", "Melody", "Melvin", "Elisabeth", "Brendon", "Kierra", "Devonte", "Karissa", "Roman", "Nikki", "Ismael", "Kira", "Emilio", "Peyton", "Keegan", "Delaney", "Emanuel", "Sonia", "Quinn", "Chelsie", "Oliver", "Tina", "Darryl", "Tierra", "Forrest", "Payton", "Abel", "Nadia", "Kyler", "Shawna", "Esteban", "Isabelle", "Guillermo", "Makenzie", "Felix", "Josephine", "Cooper", "Sidney", "Leonard", "Mikaela", "Kendrick", "Macy", "Corbin", "Francesca", "Peyton", "Regina", "Dale", "Destinee", "Geoffrey", "Julianna", "Stanley", "Kali", "Karl", "Bryanna", "Weston", "Lorena", "Eugene", "Kailey", "Jermaine", "Brianne", "Ernest", "Avery", "Graham", "Tracy", "Wade", "Jane", "Keenan", "Bria", "Devante", "Cameron", "Warren", "Kayleigh", "Nikolas", "Alyson", "Bryson", "Taryn", "Glenn", "Emilee", "Moises", "Larissa", "Alvin", "Kristine", "Trevon", "Cortney", "Clay", "Alaina", "Skylar", "Alice", "Jordon", "Chasity", "Kurt", "Tayler", "Rodolfo", "Paola", "Cedric", "Gianna", "Beau", "Sylvia", "Dwayne", "Paula", "Hugo", "Kari", "Harley", "Precious", "Harold", "Leticia", "Jarrod", "Kaitlynn", "Devan", "Nicolette", "Tyrell", "Brittani", "Isiah", "Joy", "Denzel", "Shanice", "Dane", "Kristy", "Gilbert", "Robyn", "Sam", "Clara", "Dawson", "Harley", "Stefan", "Tania", "Ray", "India", "Harry", "Mariana", "Allan", "Giselle", "Joey", "Cheyanne", "Felipe", "Lucy", "Kenny", "Jeanette", "Lukas", "Frances", "Noel", "Josie", "Elliot", "Johanna", "Brayden", "Michele", "Alfonso", "Christy", "Kristian", "Noelle", "Jarred", "Alma", "Leon", "Shania", "Tyson", "Haylee", "Davis", "Allie", "Chris", "Kaila", "Ali", "Breana", "Braxton", "Carley", "Gilberto", "Irene", "Darian", "Daniella", "Rodrigo", "Jaime", "Brody", "Shayna", "Donte", "Simone", "Deshawn", "Marilyn", "Rudy", "Randi", "Javon", "Marlene", "Keaton", "Donna", "Elliott", "Angelique", "Khalil", "Abbey", "Jayson", "Ciera", "Charlie", "Elisa", "Sheldon", "Iris", "Reid", "Yvette", "Payton", "Kristi", "Rogelio", "Talia", "Davon", "Nora", "Grayson", "Christa", "Leo", "Skyler", "Tristen", "Kourtney", "Alfred", "Elaine", "Sterling", "Rachelle", "Rashad", "Ericka", "Ramiro", "Meaghan", "Marquise", "Hunter", "Ariel", "Misty", "Blaine", "Kiera", "Nathanael", "Stefanie", "Justice", "Katharine", "Tylor", "Trisha", "Cade", "Nia", "Marlon", "Kaleigh", "Jace", "Maritza", "Clarence", "Kenya", "Mackenzie", "Cristal", "Shaquille", "Tasha", "Lewis", "Carina", "Julius", "Corinne", "Tomas", "Maribel", "Darnell", "Blanca", "Garret", "Chantel", "Reed", "Julianne", "Ralph", "Ivy", "Tyree", "Tiffani", "Brad", "Judith", "Stuart", "Paris", "Stephan", "Kelsi", "Jaquan", "Alanna", "Roderick", "Haleigh", "Clifford", "Janae", "Daquan", "Tatyana", "Santiago", "Lauryn", "Alonzo", "Lizbeth", "Demarcus", "Yvonne", "Barry", "Charity", "Duncan", "Lori", "Quincy", "Carol", "Antoine", "Kelley", "Bernard", "Kiersten", "Noe", "Yasmin", "Wilson", "Diane", "Kelly", "Kaley", "Kurtis", "Eleanor", "Mauricio", "Aliyah", "Rolando", "Margarita", "Kory", "Julissa", "Bret", "Leanna", "Jarod", "Lyndsey", "Jaden", "Viviana", "Clint", "Jill", "Damion", "Sandy", "Kirk", "Chanel", "Lamar", "Eliza", "Howard", "Hallie", "Ahmad", "Shaina", "Earl", "Tianna", "Adan", "Mollie", "Dexter", "Sheila", "Derick", "Lesley", "Dorian", "Yolanda", "Shannon", "Emilie", "Maximilian", "Paulina", "Sidney", "Kaylin", "Deonte", "Desirae", "Nigel", "Casandra", "Efrain", "Moriah", "Colten", "Tanisha", "Jarvis", "Perla", "Ezekiel", "Kaylie", "Austen", "Anita", "Walker", "Genevieve", "Deon", "Tess", "Humberto", "Stephany", "Amir", "Maranda", "Jaylen", "Justice", "Darien", "Edith", "Mohammad", "Fatima", "Perry", "Shana", "Moses", "Alisa", "Carlton", "Lena", "Gordon", "Bonnie", "Courtney", "Beatriz", "Malachi", "Georgia", "Daryl", "Kassidy", "Deangelo", "Laurel", "Leroy", "Cayla", "Issac", "Micaela", "Kasey", "Lacy", "Markus", "Kathy", "Jaron", "Kacie", "Quintin", "Baylee", "Jamar", "Alina", "Ben", "Dawn", "Fredrick", "Antoinette", "Uriel", "Bridgette", "Dwight", "Roxanne", "Norman", "Yasmine", "Holden", "Alexandrea", "Sawyer", "Carlie", "Brice", "Chandler", "Chaz", "Rochelle", "Solomon", "Heaven", "Vicente", "Suzanne", "Alvaro", "Darian", "Korey", "Ava", "Deven", "Dorothy", "Osvaldo", "Araceli", "Mike", "Selina", "Shelby", "Sally", "Stephon", "Tyra", "Freddy", "Susana", "Addison", "Antonia", "Dion", "Alysha", "Brenton", "Katelin", "Heath", "Eileen", "Cullen", "Norma", "Mitchel", "Breanne", "Leonel", "Noemi", "Terence", "Iesha", "Salvatore", "Kaylyn", "Kareem", "Shauna", "Neal", "Rosemary", "Dashawn", "Keri", "Toby", "Celina", "Roland", "Ryan", "Glen", "Aisha", "Frankie", "Luz", "Joaquin", "Clare", "Coleman", "Lucia", "Aron", "Janice", "Mohammed", "Reagan", "Bennett", "Joyce", "Everett", "Mara", "Nathanial", "Hilary", "Irvin", "Charlene", "Shayne", "Rocio", "Jean", "Skye", "Baby", "Rhiannon", "Raheem", "Lea", "Clifton", "Nathalie", "Raphael", "Gillian", "Tre", "Jacklyn", "Jayden", "Ella", "Will", "Regan", "Triston", "Kiley", "Coty", "Aurora", "Darrin", "Shirley", "Kent", "Karli", "Lonnie", "Karly", "Cornelius", "Ashlie", "Reece", "Juanita", "Milton", "Keisha", "Conrad", "Angie", "Agustin", "Blair", "Kaden", "Mariela", "Mohamed", "Maegan", "Ezra", "Mckayla", "Darrius", "Celia", "Vernon", "Annette", "Marcel", "Loren", "Caden", "Trinity", "Reynaldo", "Yadira", "Fred", "Thalia", "Tate", "Reyna", "Dandre", "Katy", "Adolfo", "Eden", "Lamont", "Alex", "Tracy", "Ashlynn", "Jamel", "Madalyn", "Gunnar", "Shelbi", "Infant", "Leanne", "Rory", "Jodi", "Lloyd", "Hailee", "Clark", "Katarina", "Arnold", "Abbie", "Rigoberto", "Kerri", "Ignacio", "Essence", "Ahmed", "Tonya", "Rickey", "Sonya", "Cortez", "Jaqueline", "Darin", "Mya", "Don", "Darlene", "Kai", "Beverly", "Darion", "Keely", "Guadalupe", "Kasandra", "Duane", "Cora", "Kristofer", "Lizette", "Camron", "Katlin", "Travon", "Makenna", "Nestor", "Jena", "Juwan", "Shantel", "Deion", "Karlee", "Rick", "Phoebe", "Dillan", "Lexus", "Ezequiel", "Maura", "Sage", "Aileen", "Jonas", "Halie", "Jasper", "Dulce", "Kolby", "Sarai", "Anton", "Jesse", "Winston", "Elaina", "Jamison", "Rita", "Kerry", "Ashly", "German", "Ellie", "Antwan", "Micah", "Jovan", "Shanna", "Brennen", "Athena", "Tristin", "Arlene", "Dimitri", "Kacey", "Camden", "Kelsea", "Trace", "Rylee", "Herbert", "Ayanna", "Ulises", "Connie", "Pierce", "Justina", "Guy", "Silvia", "Gerard", "Tammy", "Brooks", "Alayna", "Thaddeus", "Kirstin", "Pierre", "Latoya", "Johnnie", "Sade", "Jefferson", "Rebeca", "Jameson", "Gretchen", "Isaias", "Beth", "Reese", "Melina", "Kobe", "Janessa", "Cordell", "Anissa", "Robin", "Camryn", "Jimmie", "Jenifer", "Houston", "Karlie", "Freddie", "Bailee", "Reuben", "Lara", "Bradford", "Kerry", "Cyrus", "Dallas", "Kade", "Brook", "Jairo", "Octavia", "Paris", "Joanne", "Elvis", "Shaniqua", "Tobias", "Elyse", "Moshe", "Addison", "Sammy", "Cheryl", "Estevan", "Gwendolyn", "Brandan", "Alycia", "Davion", "Kailee", "Colt", "Jayla", "Kolton", "Jana", "Davonte", "Khadijah", "Donnie", "Debra", "Ladarius", "Fiona", "Trever", "Savanah", "Rhett", "Ali", "Santos", "Kori", "Kane", "Yessenia", "August", "Brittni", "Madison", "Corina", "Octavio", "Constance", "Bo", "Lexi", "Heriberto", "Mandy", "Cruz", "Carli", "Rashawn", "Christie", "Donavan", "Elissa", "Zechariah", "Christiana", "Keon", "Shelbie", "Nico", "Carlee", "Gino", "Allyssa", "Dayton", "Kaela", "Akeem", "Tabatha", "Jerrod", "Fabiola", "Hassan", "Dianna", "Jackie", "Kalyn", "Dangelo", "Mindy", "Hakeem", "Shakira", "Brandyn", "Shyanne", "Aldo", "Nikita", "Jaylon", "Alysia", "Elmer", "Kaci", "Aubrey", "Latasha", "Garrison", "Kristie", "Cristopher", "Shea", "Tyrese", "Maddison", "Dewayne", "Gladys", "Bradly", "Jalisa", "Bernardo", "Stacie", "Kellen", "Montana", "Nick", "Hollie", "Kieran", "Nadine", "Dequan", "Maricela", "Herman", "Graciela", "Coby", "Helena", "Dylon", "Stevie", "Myron", "Halle", "Dillion", "Maureen", "Gene", "Janette", "Marquez", "Brielle", "Elisha", "Destini", "Tariq", "Janie", "Junior", "Kala", "Arron", "Demi", "Silas", "Baby", "Dana", "Ingrid", "Loren", "Audra", "Lester", "Jayme", "Zakary", "Chantal", "Alonso", "Breann", "Gonzalo", "Destiney", "River", "Alessandra", "Raymundo", "Daphne", "Rico", "Bobbie", "Aric", "Tamia", "Sonny", "Jessika", "Devyn", "Leigh", "Royce", "Adrian", "Benny", "Liana", "Josh", "Kristian", "Talon", "Latisha", "Benito", "Devan", "Vance", "Macey", "Irving", "Myra", "Ellis", "Judy", "Marques", "Nataly", "Hugh", "Sage", "Asa", "Jennie", "Josef", "Cecelia", "Grady", "Dayna", "Misael", "Gabriel", "Shea", "Lucero", "Leland", "Aspen", "Jaleel", "Magdalena", "Jeff", "Joelle", "Jabari", "Ayla", "Jamarcus", "Devyn", "Ulysses", "Jami", "Jamil", "Kallie", "Bronson", "Princess", "Leslie", "Berenice", "Fidel", "Brandie", "Quinten", "Cori", "Efren", "Christen", "Stewart", "Leann", "Vaughn", "Alysa", "Dontae", "Racheal", "Mikel", "Ayana", "Devonta", "Staci", "Dan", "Elisha", "Lincoln", "Kaycee", "Mateo", "Kristyn", "Tyron", "Kortney", "Jacoby", "Jackie", "Demarco", "Terri", "Julien", "Tracey", "Ari", "Darby", "Floyd", "Itzel", "Antony", "Bobbi", "Anderson", "Kayley", "Emmett", "Martina", "Alexandro", "Krysta", "Dusty", "Shelly", "Rocky", "Catalina", "Dario", "Lacie", "Barrett", "Kalie", "Remington", "Emilia", "Rex", "Chaya", "Kevon", "Karley", "Titus", "Elsa", "Keven", "Bryana", "Austyn", "Noel", "Hernan", "Domonique", "Codey", "Iliana", "Jaylin", "Asha", "Clyde", "Tatum", "Orion", "Juana", "Alton", "Monika", "Davin", "Celine", "Donnell", "Joselyn", "Jorden", "Myranda", "Axel", "Dalia", "Hans", "Annika", "Ibrahim", "Damaris", "Kadeem", "Marcella", "Chaim", "Kianna", "Asher", "Lesly", "Garett", "Jaimie", "Jade", "Marisela", "Muhammad", "Melisa", "Broderick", "Giovanna", "Carlo", "Britany", "Gregorio", "Betty", "Andreas", "Emerald", "Raekwon", "Kailyn", "Deshaun", "Lexie", "Lionel", "Carson", "Maximillian", "Eliana", "Cecil", "Kirstie", "Erich", "Patrice", "Aiden", "Kaylynn", "Javonte", "Infant", "Rasheed", "Sydnee", "Dejuan", "Beatrice", "Edgardo", "Zoey", "Adonis", "Brittanie", "Niko", "Breonna", "Denver", "Marlena", "Jim", "Hali", "Jamaal", "Lynn", "Dakotah", "Malia", "Abram", "Roxana", "Justyn", "Kia", "Keanu", "Alecia", "Jacques", "Scarlett", "Justus", "Belinda", "Brant", "Lourdes", "Chadwick", "Jazmyn", "Augustus", "Mariam", "Montana", "Terra", "Zion", "Pauline", "Jessy", "Emely", "Harvey", "Christin", "Blair", "Ivette", "Cale", "Eboni", "Tyshawn", "Lyric", "Bryon", "Rikki", "Dallin", "Delia", "Amos", "Brooklynn", "Deondre", "Dina", "Valentin", "Macie", "Darrion", "Aubree", "Kenton", "Sherry", "Erin", "Leila", "Kole", "Alesha", "Layne", "Brionna", "Eliseo", "Billie", "Randolph", "Griselda", "Codie", "Ashanti", "Armani", "Kayli", "Giancarlo", "Tricia", "Stone", "Hana", "Ted", "Aleah", "Tristian", "Aja", "Ryne", "Elyssa", "Gunner", "Yasmeen", "Alexandre", "Traci", "Najee", "Miracle", "Domenic", "Mikala", "Ervin", "Alena", "Unknown", "Anjelica", "Kelsey", "Danica", "Tory", "Isis", "Kelton", "Susanna", "Tyquan", "Estefania", "Eddy", "Drew", "Jess", "Jocelyne", "Destin", "Sienna", "Kirby", "Lilly", "Khalid", "Janay", "Cristobal", "Katlynn", "Francesco", "Lorraine", "Hudson", "Lizeth", "Wendell", "Mireya", "Isidro", "Kimberlee", "Jody", "Abbigail", "Darrian", "Kendal", "Braeden", "Layla", "Rusty", "Shyann", "Galen", "Blake", "Lyle", "Ashely", "Chester", "Jailene", "Keshawn", "Leandra", "Abdul", "Corey", "Tavon", "Katerina", "Easton", "Kaelyn", "Justen", "Kenia", "Marcelo", "Yazmin", "Dionte", "Sheena", "Turner", "Kenzie", "Maverick", "Sarina", "Storm", "Stormy", "Daron", "Kamryn", "Shelton", "Leilani", "Mickey", "Anika", "Syed", "Lana", "Edmund", "Esperanza", "Spenser", "Maira", "Devontae", "Annabelle", "Emerson", "Ashli", "Kennedy", "Chelsi", "Auston", "Gracie", "Samir", "Jessi", "Greg", "Kinsey", "Tom", "Eryn", "Forest", "Marley", "Markel", "Irma", "Braydon", "Leeann", "Rohan", "Reina", "Trae", "Jerrica", "Romeo", "Haylie", "Dominque", "Kimberley", "Tayler", "Tiera", "Demario", "Tiffanie", "Kegan", "Chyna", "Kelby", "Zaria", "Jerrell", "Maia", "Ryder", "Chantelle", "Greyson", "Juliette", "Keyshawn", "Kristal", "Alden", "Amari", "Brayan", "Amie", "Antwon", "Dasia", "Shamar", "Unique", "Draven", "Olga", "Koby", "Nayeli", "Jerod", "Juliet", "Pete", "Rhonda", "Rodrick", "Alivia", "Ashley", "Marlee", "Darrien", "Kyleigh", "Deontae", "Tyesha", "Otis", "Sydni", "Shaquan", "Patience", "Randal", "Krystle", "Nasir", "Tatianna", "Scotty", "Janine", "Wilfredo", "Valentina", "Gianni", "Daisha", "Sherman", "Kathrine", "Darwin", "Laurie", "Seamus", "Sydnie", "Genaro", "Jamila", "Samson", "Mattie", "Morris", "Jodie", "Jaxon", "Jasmyn", "Markell", "Tamera", "Ronaldo", "Madyson", "Waylon", "Anahi", "Brennon", "Laken", "Derik", "Lakeisha"]
clue = 3
good_clue = 4
confident = 6
slam_dunk = 20

locPrep = ["in", "outside", "on", "between", "at", "beside", "by", "beyond", "near", "in front of", "nearby", "in back of", "above", "behind", "below", "next to", "over", "on top of", "under", "within", "up", "beneath", "down", "underneath", "around", "among", "through", "along", "inside", "against"]
def remstwords(str):
    t_str = copy.deepcopy(str)
    for w in t_str:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or string.punctuation:
            str.remove(w)
    return str

def checkStopWord(word):
    word_lm = lemmatizer.lemmatize(word.lower())
    if word_lm in STOPWORDS:
        return True
    return False

def PosTag(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def WordMatch(ques, sent):
    score = 0
    ques_tag = PosTag(ques)
    for q in ques_tag:
        if not checkStopWord(q[0]):
            if q[1] == "VB" or q[1] == "VBD" or q[1] == "VBG" or q[1] == "VBN" or q[1] == "VBP" or q[1] == "VBZ":
                w = ps.stem(q[0].lower())
                for s in sent.split():
                    s_lm = ps.stem(s.lower())
                    if w == s_lm:
                        score = score + 6
            else:
                w = ps.stem(q[0].lower())
                for s in sent.split():
                    s_lm = ps.stem(s.lower())
                    if w == s_lm:
                        score = score + 3
    return score

def containsNER(q,category):
    q_tokens  = nltk.word_tokenize(q)
    q_tag = nltk.pos_tag(q_tokens)
    ne_tag = nltk.ne_chunk(q_tag)
    for tree in ne_tag.subtrees():
        if tree.label() == category:
            return True
    return False

def contains(L,string):
    for q in L.split():
        if q == string:
            return True
    return False

def containsPOSTag(L,string):
    L_tag = PosTag(L)
    s = [True if x[1] == string else False for x in L_tag]
    return s

def containsList(L,lt):
    for q in L.split():
        if q in lt:
            return True
    return False

def containsList_lemma(L,lt):
    q_lm = ps.stem(L.lower())
    for q in q_lm.split():
        if q in lt:
            return True
    return False

def containsPP(q,string):
    q_tag = PosTag(q)
    for i,w1 in enumerate(q.split()):
        if w1 == string:
            if q_tag[i+1][0] == "IN":
                return True
    return False

def containsNPwithPP(s):
    q_tag = PosTag(s)
    proper_noun = [x[0] for x in q_tag if x[1] == "NNP" or x[1] == "NNPS"]
    containsPP(s,proper_noun)

def whoRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if (not containsNER(q,"PERSON") or not containsList(q, OCCUPATION) or not containsList(q, PERSON_NAMES)) and (containsNER(s,"PERSON") or containsList(s,OCCUPATION) or containsList(s,PERSON_NAMES)):
        score = score + confident
    if (not containsNER(q,"PERSON") or not containsList(q, OCCUPATION) or not containsList(q, PERSON_NAMES)) and contains(s,"name"):
        score = score + good_clue
    if containsPOSTag(s,"NNP") or containsPOSTag(s, "NNPS"):
        score= score + good_clue
    return score

def whatRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(q, MONTH) and containsList(s, ["today", "yesterday", "tomorrow", "last night"]):
        score = score + clue
    if contains(q,"kind") and containsList_lemma(s, ["call", "from"]):
        score = score + good_clue
    if contains(q,"name") and containsList_lemma(s, ["name", "call", "known"]):
        score = score + slam_dunk
    if containsPP(q, "name") and containsNPwithPP(s):
        score =  score + slam_dunk
    return score
year_list = []
for i in range(1400,2000):
    year_list.append(str(i))
def whenRule(q,s):
    score = 0
    if containsList(s, TIME) or containsList(s, year_list) :
        score = score + good_clue
        score = score + WordMatch(q,s)
    if contains(q,"the last") and containsList(s, ["first", "last", "since", "ago"]):
        score = score + slam_dunk
    if containsList(q,["start", "begin"]) and containsList(s, ["start", "begin", "since", "year"]):
        score = score + slam_dunk
    return score

def whereRule(q,s):
    score = 0
    score = score + WordMatch(q,s)
    if containsList(s, locPrep):
        score = score + good_clue
    if containsNER(s, "LOCATION") or containsList(s, LOCATION) or containsNER(s, "GPE"):
        score = score + confident
    return score

def whyRule(q,s, best, prev_sent, next_sent):
    score = 0
    for b_sent in best:
        if s == b_sent:#doubt if in works
            score = score + good_clue
        if  next_sent == b_sent:
            score = score + clue
        if prev_sent == b_sent:
            score = score + clue
    if contains(s, "want"):
        score = score + good_clue
    if containsList(s, ["so", "because"]):
        score = score + good_clue
    return score

def datelineRule(q):
    score = 0
    if contains(q, "happen"):
        score = score + good_clue
    if contains(q, "take") and contains(q, "place"):
        score = score + good_clue
    if contains(q, "this"):
        score = score + slam_dunk
    if contains(q, "story"):
        score = score + slam_dunk
    return score

def convertSenttolist(s):
    sent_li = []
    for word in s.split():
        sent_li.append(word)
    return sent_li

def findWH(sent):
    for word in reversed(sent.split()):
        if ques_words.__contains__(word.lower()):
            return word.lower()

os.chdir(sys.argv[1])
o = open('../out.txt', 'w')
files = sorted(glob.glob('*.questions'))
cnt = 0
for file in files:
    cnt = cnt + 1
    print(cnt)
    fileId = file.title().split('.')[0]
    ques_list = []
    qid_list = []
    with open(file, 'r') as g:
        ques_lines = g.readlines()
        for line in ques_lines:
            if line.__contains__("QuestionID:"):
                qid_list.append(line)
            if line.__contains__("Question:"):
                ques = line.split("Question:")[1]
                ques_list.append(ques)
    with open(fileId + '.story', 'r') as f:
        story_text = f.read()
        header = story_text.split("TEXT:")[0]
        date = None
        if header.__contains__("DATE: "):
            dateline = header.split("DATE:")[1]
            date = dateline.split("\n")[0]
        #print date
        sent_list = nltk.sent_tokenize(story_text.split("TEXT:")[1])
    for z,q in enumerate(ques_list):
        question_type = ""
        for x in reversed(q.split()):
            if x.lower() == "who":
                question_type="who"
                break
            if x.lower() == "what":
                question_type="what"
                break
            if x.lower() == "when":
                question_type="when"
                break
            if x.lower() == "where":
                question_type="where"
                break
            if x.lower() == "why":
                question_type="why"
                break
        list1 = []
        dateline_score = 0
        if question_type=="who":
            for s in sent_list:
                score = whoRule(q,s)
                list1.append((s,score))
        elif question_type=="what":
            for s in sent_list:
                score = whatRule(q,s)
                list1.append((s,score))
        elif question_type=="when":
            dateline_score = datelineRule(q)
            for s in sent_list:
                score = whenRule(q,s)
                list1.append((s,score))
        elif question_type=="where":
            dateline_score = datelineRule(q)
            for s in sent_list:
                score = whereRule(q,s)
                list1.append((s,score))
        elif question_type=="why":
            score = []
            best = []
            for k,s in enumerate(sent_list):
                score.append(WordMatch(q,s))
                best.append((s, score[k]))
            best_list = sorted(best, key=lambda  x: (-x[1],x[0]))
            sent_best = []
            for j in range(10):
                sent_best.append(best_list[j][0])
            for i,s in enumerate(sent_list):
                if i == 0:
                    score = whyRule(q,s,sent_best,None,sent_list[i+1])
                elif  i == len(sent_list) - 1:
                    score = whyRule(q,s,sent_best,sent_list[i-1],None)
                else:
                    score = whyRule(q,s,sent_best,sent_list[i-1],sent_list[i+1])
                list1.append((s,score))
        else:
            for s in sent_list:
                score = WordMatch(q,s)
                list1.append((s,score))
        max_s = max(list1, key=lambda x:x[1])
        s = ""
        for x in list1:
            if question_type == "why":
                if x[1] == max_s[1]:
                    s = x[0]
            else:
                if x[1] == max_s[1]:
                    s = x[0]
                    break
        if question_type == "when" or question_type == "where":
            if dateline_score > max_s[1]:
                s = date
        if max_s[1] == 0:
            if question_type == "when" or question_type == "where":
                s = date
            elif question_type == "why":
                s = sent_list[len(sent_list)-1]
            else:
                s = sent_list[0]
        # Tie rule
        # list1 contains the sentences with scores
        #s_li = convertSenttolist(s)
        #s_li_copy = copy.deepcopy(s_li)
        #for word in s_li_copy:
            #if checkStopWord(word):
                #s_li.remove(word)
        #new_str = ""
        #for word in s_li:
            #new_str = new_str + word + " "
        s1 = s.strip()
        s2 = s1.replace("\n"," ")

        #call our code on q
        # first find the answer type
        new_ans = []
        Whword = findWH(q)
        matched_ans = []
        if Whword == "who" or Whword == "where":
            anstype = ans_type[Whword]
            ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2)))
            for tree in ner_tag.subtrees():
                if anstype == "location":
                    if tree.label().lower() == anstype or tree.label().lower() == "gpe":
                        matched_ans.append(tree.leaves()[0][0])
                else:
                    if tree.label().lower() == anstype:
                        matched_ans.append(tree.leaves()[0][0])
        s2_list = s2.split()
        if Whword == "when":
            ner_tag = st.tag(s2_list)
            matched_ans = []
            for tree in ner_tag:
                if tree[1].lower() == "date" or tree[1].lower() == "time":
                    matched_ans.append(tree[0])
            if matched_ans == []:
                for word in s2_list:
                    if word in TIME1 or word in year_list:
                        matched_ans.append(word)
        if Whword == "how":
            q_list = q.split()
            for i,word in enumerate(q_list):
                if word.lower() == "how":
                    if word.lower() + " " + q_list[i+1] in ans_type:
                        ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2)))
                        matched_ans = []
                        temp_list = []
                        for tree in ner_tag.subtrees():
                            if tree.label().lower() == "date" or tree.label().lower() == "time":
                                temp_list.append(tree.leaves[0][0])
                        for y in ner_tag.leaves():
                            if y[1] == "CD":
                                if not y[0] in x:
                                    matched_ans.append(y[0])
        if Whword == "what":
            ner_tag = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(s2)))
            matched_ans = []
            for tree in ner_tag.subtrees():
                if tree.label().lower() == "organization":
                    matched_ans.append(tree.leaves()[0][0])
        for a,words in enumerate(s2_list):
            for ans in matched_ans:
                new_ans_str = ""
                if words == ans:
                    if a == 0:
                        new_ans_str = new_ans_str + " " + words + " " + s2_list[a+1]
                    elif a == len(s2_list) - 1:
                        new_ans_str = new_ans_str + " " + s2_list[a-1] + " " + words
                    else:
                        new_ans_str = new_ans_str + " " + s2_list[a-1] + " " + words + " " + s2_list[a+1]
                    new_ans.append(new_ans_str)
        # Prcess the s1.
        # Put if loop of answertype so that it seasy to debug. Currently who and where.
        # Put ner on s2 for the answer type
        # print 5 words before and and after save the value on file.
        value = ""
        if len(new_ans) > 0:
            value = qid_list[z] + "Answer:"
            for ans in new_ans:
                ans = " ".join("".join([" " if ch in string.punctuation else ch for ch in ans]).split())
                value = value + " " +ans
            value = value + "\n\n"
        else:
            s2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in s2]).split())
            value = qid_list[z] + "Answer: " + s2 + "\n\n"
        #value = " ".join("".join([" " if ch in string.punctuation else ch for ch in value]).split())
        #value = value.translate(None, string.punctuation)
        o.write(value)
        sorted_list = sorted(list1, key=lambda  x: (-x[1],x[0]))
o.close()

'''def Postagger(sent):
    sent_tag = nltk.pos_tag(nltk.word_tokenize(sent))
    return sent_tag

def findWH(sent):
    for word in reverse(sent).split():
        if ques_words.__contains__(word.lower()):
            return word

def findNNP(sent_tagged):
    #i = [x for x in sent_tagged if x[1] == "NNP" or x[1] == "NNPS"]
    nnp = []
    flag = 0
    s = None
    for x in sent_tagged:
        if x[1] == "NNP" and flag == 0:
            s = x[0]
            flag = 1
        elif x[1] == "NNP" and flag == 1:
            s = s + " " + x[0]
        else:
            if s != None:
                nnp.append(s)
            s = None
            flag = 0
    if sent_tagged[-1][1] == "NNP":
        nnp.append(s)
    #for x in i:
        #sent_tagged.remove(x)
    return nnp

def removestopwords_punct(ques):
    t_ques = copy.deepcopy(ques)
    for w in t_ques:
        word_lm = lemmatizer.lemmatize(w[0].lower())
        if word_lm in STOPWORDS or word_lm in string.punctuation:
            ques.remove(w)
    return ques

def findcomplNominal(tag_sent):

    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'JJ':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findotherNominal(tag_sent):
    nom = []
    for i,(x,t) in enumerate(tag_sent):
        if tag_sent[i][1] == 'NN' and tag_sent[i-1][1] == 'NN':
            nom.append(tag_sent[i-1][0] + " " + tag_sent[i][0])
            tag_sent.remove(tag_sent[i-1])
            tag_sent.remove((x,t))
    return nom

def findnounAdj(tag_sent):
    na = []
    i = [tag_sent.index(x) for x in tag_sent if x[1] == "JJ"]
    for z in i:
        noun = [y[0] for y in tag_sent if tag_sent.index(y) > z and y[1] == "NN" or y[1] == "NNS"]
        na.append(tag_sent[z][0] + " " + ' '.join(map(str, noun)))
    return na

def findallNoun(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "NN" or x[1] == "NNS"]
    return i

def findallVerb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "VBZ" or x[1] == "VB" or x[1] == "VBD" or x[1] == "VBG" or x[1] == "VBN" or x[1] == "VBP"]
    return i

def findallAdverb(tag_sent):
    i = [x[0] for x in tag_sent if x[1] == "RB" or x[1] == "RBR" or x[1] == "RBS"]
    return i

def countexp_nnp(para, nnp):
    cnt = 0
    for stxt in nnp:
        if stxt in para:
            cnt = cnt + 1
            break
        else:
            for w in stxt.split():
                if w in para:
                    cnt = cnt + 1
                    break
    return cnt

def countexp(para, searchText):
    cnt = 0
    for stxt in searchText:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
    return cnt

def countexp_verb(para, searchText):
    cnt = 0
    searchWords = [lemmatizer.lemmatize(s) for s in searchText]
    for stxt in searchWords:
        if stxt.lower() in para.lower():
            cnt = cnt + 1
    return cnt

def countexp_na(para, searchText):
    cnt = 0
    flag = 0
    for stxt in searchText:
        for sen in nltk.sent_tokenize(para):
            for word in stxt.split():
                if word in sen:
                    flag = 1
                else:
                    flag = 0
            if flag == 1:
                cnt = cnt + 1
                break
    return cnt


os.chdir(sys.argv[1])
files = glob.glob('*.questions')
for file in files:
    ans_types = []
    fileId = file.title().split('.')[0]
    quotations = []
    ner_tag = []
    nnp_words = []
    complNominal = []
    otherNominal = []
    nounAdj = []
    otherNoun = []
    verb = []
    adverb = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.__contains__("Question:"):
                ques = line.split("Question:")
                #print "\n" + ques[1]
                Wh_word = findWH(ques[1])
                #print Wh_word
                q_tagged = Postagger(ques[1])
                q_tagged_copy = copy.deepcopy(q_tagged)
                #print "Answer Type: "
                if Wh_word.lower() == "how":
                    i = [x[0] for x in q_tagged].index(Wh_word)
                    if q_tagged[i+1][1].lower() == 'jj':
                        ans_types.append(ans_type[Wh_word.lower() + " jj"])
                    elif q_tagged[i+1][1].lower() == "mod":
                        ans_types.append(ans_type[Wh_word.lower() + " mod"])
                    elif q_tagged[i+1][1].lower() == "long":
                        ans_types.append(ans_type[Wh_word.lower() + " long"])
                    elif q_tagged[i+1][1].lower() == "far":
                        ans_types.append(ans_type[Wh_word.lower() + " far"])
                    elif q_tagged[i+1][1].lower() == "much":
                        ans_types.append(ans_type[Wh_word.lower() + " much"])
                    elif q_tagged[i+1][1].lower() == "old":
                        ans_types.append(ans_type[Wh_word.lower() + " old"])
                    elif q_tagged[i+1][1].lower() == "often":
                        ans_types.append(ans_type[Wh_word.lower() + " often"])
                    else:
                        ans_types.append(ans_type[Wh_word.lower()])
                else :
                    ans_types.append(ans_type[Wh_word.lower()])
                #findNominal(q_tagged)
                #ner_tag.append(st.tag(ques[1].split()))
                #print ner_tag
                quot = []
                for val in quoted.findall(ques[1]):
                    if val:
                        quot.append(val)
                quotations.append(quot)
                #print quotations
                nnp_words.append(findNNP(q_tagged))
                complNominal.append(findcomplNominal(q_tagged))
                otherNominal.append(findotherNominal(q_tagged))
                nounAdj.append(findnounAdj(q_tagged))
                otherNoun.append(findallNoun(q_tagged))
                removestopwords_punct(q_tagged)
                verb.append(findallVerb(q_tagged))
                adverb.append(findallAdverb(q_tagged))
    #print fileId
    #print nnp_words
    with open(fileId + '.story', 'r') as g:
        story = g.read().split("TEXT:")[1]
        paras = re.split("[\.|\"|!]\s*\n\n+", story)
        #sents = nltk.sent_tokenize(story)

        #for each question
        for i in range(len(quotations)):
            count_list = []
            for para in paras:
                count = 0
                count = count + countexp(para, quotations[i])
                count = count + countexp_nnp(para, nnp_words[i])
                count = count + countexp(para, complNominal[i])
                count = count + countexp(para, otherNominal[i])
                count = count + countexp_na(para, nounAdj[i])
                count = count + countexp(para, otherNoun[i])
                count = count + countexp_verb(para, verb[i])
                count = count + countexp(para, adverb[i])
                count_list.append(count)
                if count > 0:
                    ner_tagged = st.tag(para.split())
                    matched_words = [x[0] for x in ner_tagged if x[1].lower() in ans_types[i]]'''
                    #print ner_tagged
                    #print matched_words
                    # we matched the words in the tagged para to the ans types.
                    #print "\n"
            #print "QUES " + (i+1).__str__() + ": "
            #print count_list

            #print "\n"
        #for sent in sents:
            #print quotations
