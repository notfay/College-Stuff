package model;

import mediator.Mediator;

public abstract class Kendaraan {
	
	String namaMobil;
	Mediator mediator;
	
	public Kendaraan(String namaMobil, Mediator mediator) {
		super();
		this.namaMobil = namaMobil;
		this.mediator = mediator;
	}
	
	
	public abstract void diPersimpangan();
	
	
}
