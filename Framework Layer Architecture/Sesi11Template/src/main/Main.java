package main;

import model.Firefighter;
import model.Satpam;
import model.Worker;

public class Main {

	public static void main(String[] args) {
		Worker pemadam = new Firefighter("Jamal", 50);
		Worker satpam = new Satpam("Bagas", 20);
		
		pemadam.dailyRoutine();
		System.out.println("");
		satpam.dailyRoutine();

	}

}
