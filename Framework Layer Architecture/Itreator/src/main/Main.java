package main;

import java.util.ArrayList;

import iterator.FIFO;
import iterator.Iterator;
import iterator.LIFO;
import model.Mobil;

public class Main {

	public static void main(String[] args) {
		
		
		ArrayList<Mobil> listFifo = new ArrayList<>();
		listFifo.add(new Mobil("Senia", 2005));
		listFifo.add(new Mobil("Yoyo", 2009));
		
		Iterator<Mobil> fifo = new FIFO<>(listFifo);
		
		while(fifo.hasNext()) {
			System.out.println(fifo.getNext().getMerek());
		}
		
		
		ArrayList<Mobil> listLifo = new ArrayList<>();
		listLifo.add(new Mobil("WOOO", 2005));
		listLifo.add(new Mobil("COO", 2009));
		
		Iterator<Mobil> lifo = new LIFO<>(listLifo);
		

		while(lifo.hasNext()) {
			System.out.println(lifo.getNext().getMerek());
		}
		
	}

}
