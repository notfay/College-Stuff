package model;
import mediator.Mediator;

public class Mobil extends Kendaraan {

	public Mobil(String namaMobil, Mediator mediator) {
		super(namaMobil, mediator);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void diPersimpangan() {
		// TODO Auto-generated method stub
		System.out.println(namaMobil + "Sampai di persimpangan");
		mediator.mauNyebrang(namaMobil);
	}
	
}
