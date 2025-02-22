package main;

import builder.Director;
import builder.FamilyCarBuilder;
import builder.SportsCarBuilder;
import model.car.FamilyCar;
import model.car.SportsCar;

public class Main {

	public Main() {
		
		FamilyCarBuilder familyBuilder = new FamilyCarBuilder();
		Director director = new Director(familyBuilder);
		director.buildFamilyCar();
		
		FamilyCar familyCar = familyBuilder.getCar();
		System.out.println(familyCar.getEngine().getName());
		
		
		
		
		SportsCarBuilder sportsBuilder = new SportsCarBuilder();
		director = new Director(sportsBuilder);	
		director.buildSportCart();
		
		SportsCar sportsCar = sportsBuilder.getCar();
		System.out.println(sportsCar.getEngine().getName());
		
		
	}

	public static void main(String[] args) {
		
		new Main();

	}

	
	
	
	
}
