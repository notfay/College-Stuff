package main;

import mediator.LampuMerah;
import model.Mobil;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LampuMerah lampuMerah = new LampuMerah();
		
		Mobil mobil1 = new Mobil("Mobiol", lampuMerah);
		Mobil mobil2 = new Mobil("adadaMobiol", lampuMerah);
		Mobil mobil3 = new Mobil("WEWEE", lampuMerah);
		
		mobil1.diPersimpangan();
		mobil2.diPersimpangan();
		mobil3.diPersimpangan();
	}

}
