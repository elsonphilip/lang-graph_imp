is_a(dog, mammal).
is_a(cat, mammal).
is_a(eagle, bird).
is_a(penguin, bird).
is_a(salmon, fish).
is_a(shark, fish).
is_a(bat, mammal).

has_trait(mammal, warm_blooded).
has_trait(bird, warm_blooded).
has_trait(bird, feathers).
has_trait(fish, cold_blooded).

eats(cat, salmon).
eats(shark, salmon).

has_property(Animal, Trait) :-
    is_a(Animal, Class),
    has_trait(Class, Trait).

is_endotherm(Animal) :-
    has_property(Animal, warm_blooded).

can_fly(Animal) :-
    is_a(Animal, bird),
    Animal \= penguin.

can_fly(bat).

is_carnivore(Animal) :-
    eats(Animal, Prey),
    is_a(Prey, _).

is_predator(Animal) :-
    is_carnivore(Animal),
    is_a(Animal, mammal).

shares_category(Animal1, Animal2) :-
    is_a(Animal1, Class),
    is_a(Animal2, Class),
    Animal1 \= Animal2.

is_aquatic_predator(Animal) :-
    is_a(Animal, fish),
    eats(Animal, _).

has_feathers(Animal) :-
    has_property(Animal, feathers).

is_cold_blooded_swimmer(Animal) :-
    is_a(Animal, fish),
    has_property(Animal, cold_blooded).
